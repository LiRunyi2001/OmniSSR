import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import sys
sys.path.append('../')
sys.path.append('./')

from multi_viewer import MultiViewer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import torchvision.utils as tvu
import cv2
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

from overlapping_tile import partion_overlapping_window, reverse_overlapping_window

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size  # pil_img.size = [w, h]
	print(f"loaded input image of size (width:{w}, height:{h}) from {path}")
	w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload"
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="./stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="./vqgan_cfw_00011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=4.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)

	parser.add_argument(
		"--dist_k_fold",
		type=str,
		default=None,
		help="Whether use distributed sampling, and the k-fold index",
	)

	opt = parser.parse_args()
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")   # UNet Model
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = model.to(device)

	model.configs = config

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.to(device)
	vq_model.decoder.fusion_w = opt.dec_w  # from args: dec_w

	with torch.no_grad():
		factor = int(opt.upscale)
		def bicubic_kernel(x, a=-0.5):
			if abs(x) <= 1:
				return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
			elif 1 < abs(x) and abs(x) < 2:
				return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
			else:
				return 0
		k = np.zeros((factor * 4))
		for i in range(factor * 4):
			x = (1 / factor)*(i - np.floor(factor*4/2) +0.5)
			k[i] = bicubic_kernel(x)
		k = k / np.sum(k)
		kernel = torch.from_numpy(k).float().cuda()
		from svd_replacement import SRConv
		H_funcs = SRConv(kernel / kernel.sum(), \
							3, 256, 'cuda', stride = factor)
		model.H_funcs = H_funcs

		H_funcs_cpu = SRConv(kernel.cpu() / kernel.sum().cpu(), \
							3, 256, 'cpu', stride = factor)

	# OmniSSR Hyperparameters #
	gamma_l = 0.5    # gamma_l for latent z interpolation, default: 0.5
	gamma_e = 1.0   # gamma_e for gradient decomposition in erp, default: 1.0
	model.gamma_l = gamma_l
	model.gamma_e = gamma_e

	# Best ERP<->TAN config #
	pre_upscale = 4  # preupsampling in OTII
	nrows = 4
	fov = (75, 75)
	patch_size = (512, 512)  # half of the gt height
	
	multi_viewer = MultiViewer()

	model.pre_upscale = pre_upscale
	model.nrows = nrows
	model.fov = fov
	# model.patch_size = (int(patch_size[0]/opt.f), int(patch_size[1]/opt.f))

	model.hr_erp_size = int(1024), int(2048)  # gt size

	if 'nrows' not in opt.outdir:
		outpath = opt.outdir + f"_gamma-latent-{gamma_l}_gamma-erp-{gamma_e}_input-size-{opt.input_size}_pre-upscale-{pre_upscale}_nrows-{nrows}_fov-{fov[0]}-{fov[1]}_patchsize-{patch_size[0]}-{patch_size[1]}"
	else:
		outpath = opt.outdir
	os.makedirs(outpath, exist_ok=True)

	batch_size = opt.n_samples  # default: 2

	img_list_ori = list(filter(lambda f: ".png" in f, os.listdir(opt.init_img)))
	img_list_ori = sorted(img_list_ori)
	if opt.dist_k_fold is not None:
		dist_k, dist_n = map(int, opt.dist_k_fold.split('/'))
		img_list_ori = list(chunk(img_list_ori, math.ceil(len(img_list_ori) / dist_n)))[dist_k-1]
		img_list_ori = list(img_list_ori)
		print(f"Using distributed sampling, k={dist_k}, n={dist_n}")

	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	lr_image_list = []
	for item in img_list_ori:
		if os.path.exists(os.path.join(outpath, item)):
			img_list.remove(item)
			continue
			
		cur_image = load_img(os.path.join(opt.init_img, item))  # load image, transform to torch.tensor [1, C, H, W] range: [-1 ~ +1]
		lr_image_list.append(cur_image)
		# max size: 1800 x 1800 for V100
		if opt.upscale > 1.0:
			cur_image = F.interpolate(
					cur_image,
					size=(int(cur_image.size(-2)*opt.upscale),
						int(cur_image.size(-1)*opt.upscale)),
					mode='bicubic', align_corners=True
					)
			# print(f"cur_image shape:{cur_image.shape}")
		init_image_list.append(cur_image)

	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000

	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext  # default: autocast
		

	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				all_samples = list()
				for n in trange(len(init_image_list), desc="Sampling"):
					tic = time.time()
					init_image = init_image_list[n]
					init_image = init_image.clamp(-1.0, 1.0)

					basename = os.path.splitext(os.path.basename(img_list[n]))[0]
					os.makedirs(f"{outpath}/sr_tan_examples/X{opt.upscale}/{basename}", exist_ok=True)
					os.makedirs(f"{outpath}/lr_tan_examples/X{opt.upscale}/{basename}", exist_ok=True)
					os.makedirs(f"{outpath}/erp_output", exist_ok=True)

					x_recon_path = f"{outpath}/x_recon/X{opt.upscale}/{basename}"
					os.makedirs(f"{outpath}/x_recon/X{opt.upscale}/{basename}", exist_ok=True)
					model.x_recon_path = x_recon_path
					model.x_recon_save_count = opt.ddpm_steps

					pers, _, _, _ = multi_viewer.equi2pers(init_image.cpu(), fov=fov, nrows=nrows, patch_size=patch_size, pre_upscale=model.pre_upscale)
					pers = pers.cuda()
					# pers of shape: [N, C, p_h, p_w, patch_num=18]
					init_pers_inputs = []
					latent_pers_inputs = []

					tan_weights = model._gaussian_weights(tile_width=pers.shape[3], tile_height=pers.shape[2], nbatches=pers.shape[0]*pers.shape[-1]).float()
					tan_weights = tan_weights[:, :pers.shape[1], :, :]
					tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np', Np=pers.shape[-1])
					model.img_tan_weights_pers = tan_weights_pers
					model.init_pers = pers
					model.lr_erp = lr_image_list[n].to(device)

					# Tangent Loop at Encoder#
					for idx in range(pers.shape[-1]):
						sub_init_image = pers[..., idx]
						# tvu.save_image(torch.clamp((sub_init_image + 1.0) / 2.0, min=0, max=1), f"{outpath}/lr_tan_examples/X{opt.upscale}/{basename}/lr_tan_{idx:02}.png")
						# print(f'>>>>>>>>>>>Tangent Loop [{idx:02}]>>>>>>>>>>>>')
						# print(sub_init_image.size())
						ori_size = None

						init_template = sub_init_image
						init_pers_inputs.append(init_template)

						init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_template))  # move to latent space, LR image input through VQGAN Encoder, and output latent z
						latent_pers_inputs.append(init_latent)

					init_pers_in = torch.stack(init_pers_inputs, dim=-1)  # [B, C, Hp, Wp, Np]

					latent_pers_in = torch.stack(latent_pers_inputs, dim=-1)  # [B, C, Hp, Wp, Np]
					model.num_tan_patch = pers.shape[-1]
					latent_pers_in = rearrange(latent_pers_in, 'B C H W Np -> (B Np) C H W')

					text_init = ['']*opt.n_samples  # n_samples (default): 2
					semantic_c = model.cond_stage_model(text_init)  # [n_samples, 77, 1024]

					noise = torch.randn_like(latent_pers_in)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					t = repeat(torch.tensor([999]), '1 -> b', b=latent_pers_in.size(0))  # shape [B*Np,]
					t = t.to(device).long()
					x_T = model.q_sample_respace(x_start=latent_pers_in, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)  # q_sample: 加噪。对 Latent LR 加噪，得到 x_T
						# x_T = noise

					samples, _ = model.sample_canvas_pano(cond=semantic_c, struct_cond=latent_pers_in, batch_size=latent_pers_in.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)

					samples_pers = rearrange(samples, "(B Np) C H W -> B C H W Np", Np=pers.shape[-1])

					final_pers_outputs = []
					for idx in range(pers.shape[-1]):
						init_template = init_pers_in[..., idx]
						samples = samples_pers[..., idx]
						_, enc_fea_lq = vq_model.encode(init_template)
						sub_x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
						# sub_x_samples = model.decode_first_stage(samples_pers[..., idx]) # if no skip in encoder and decoder

						# if ori_size is not None:
						# 	sub_x_samples = sub_x_samples[:, :, :ori_size[-2], :ori_size[-1]]
						if opt.colorfix_type == 'adain':
							sub_x_samples = adaptive_instance_normalization(sub_x_samples, init_template)  # 改进，针对tan patch做color fix
						elif opt.colorfix_type == 'wavelet':
							sub_x_samples = wavelet_reconstruction(sub_x_samples, init_template)

						final_pers_outputs.append(sub_x_samples)
						# tvu.save_image(torch.clamp((sub_x_samples + 1.0) / 2.0, min=0, max=1), f"{outpath}/sr_tan_examples/X{opt.upscale}/{basename}/sr_tan_{idx:02}.png")

					# Tangent Loop End #
					final_pers_out = torch.stack(final_pers_outputs, dim=-1)  # [B, C, H, W, Np]
					tan_weights = model._gaussian_weights(tile_width=final_pers_out.shape[3], tile_height=final_pers_out.shape[2], nbatches=final_pers_out.shape[0]*final_pers_out.shape[-1]).float()
					tan_weights = tan_weights[:, :final_pers_out.shape[1], :, :]
					tan_weights_pers = rearrange(tan_weights, '(B Np) C H W -> B C H W Np', Np=final_pers_out.shape[-1])


					torch.cuda.empty_cache()
					weighted_erp_tensor = multi_viewer.pers2equi((final_pers_out * tan_weights_pers).cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
					erp_weights = multi_viewer.pers2equi(tan_weights_pers.cpu(), fov=fov, nrows=nrows, erp_size=(1024, 2048), pre_upscale=model.pre_upscale)
					erp_tensor = weighted_erp_tensor / erp_weights
					x_samples = erp_tensor

					
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					toc = time.time()
					# print(f"Runtime per ERP: {toc - tic:.2f}s")

					for i in range(init_image.size(0)):
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(f"{outpath}/erp_output", basename+'.png'))
						# init_image = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0)
						# init_image = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
						# Image.fromarray(init_image.astype(np.uint8)).save(
						# 	os.path.join(outpath, basename+'_lq.png'))


	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		f" \nEnjoy.")
	
	return 0


if __name__ == "__main__":
	main()
