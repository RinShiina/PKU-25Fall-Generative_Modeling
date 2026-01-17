import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
import os

import torch
import asyncio
import itertools
import torchvision.transforms.functional as TF 

from inverse_stable_diffusion import *
from diffusers import DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from attackers import *
from arguments import parse_args

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import torch.fft
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# [FINAL] 256-ID Scheme: Strong Ring0 + Corner Artifact Removal (Vignette)
# ==============================================================================

def get_watermark_config_256(user_id):
    """
    User ID (0-255) -> 4 Rings configuration
    改进：Ring 0 增强，其他环保持平衡。
    """
    user_id = int(user_id) % 256
    
    # 保持瘦身后的环宽 (width=2)，以保留更多画质
    rings_spatial = [
        (11, 13),   # R0 (Low-Mid)
        (16, 18),   # R1 (Mid)
        (21, 23),   # R2 (Mid-High)
        (26, 28)    # R3 (High)
    ]
    
    # 候选频率池 (保持避开 k=3,5)
    candidate_pools = [
        [7, 9, 11, 13],    # R0
        [11, 13, 15, 17],  # R1
        [15, 17, 19, 21],  # R2
        [21, 23, 25, 27]   # R3
    ]
    
    # [关键修改] 强度权重调整
    # Ring 0: 提升至 1.2 (最容易出错，必须加强)
    # Ring 1-3: 保持 1.0 (平衡画质与检测)
    ring_scales = [1.2, 1.0, 1.0, 1.0] 
    
    config_list = []
    debug_info = [] 
    
    temp_id = user_id
    
    for i in range(4):
        state = temp_id % 4
        temp_id = temp_id // 4
        
        r_s, r_e = rings_spatial[i]
        pool = candidate_pools[i]
        target_k = pool[state]
        
        config_list.append({
            'k': target_k, 
            'r_start': r_s, 
            'r_end': r_e,
            'scale': ring_scales[i] # 携带权重信息
        })
            
        debug_info.append({
            'ring_idx': i,
            'range': (r_s, r_e),
            'state': state,
            'pool': pool,
            'target_k': target_k
        })
        
    return config_list, debug_info

def get_hybrid_pattern(latents_shape, device, config_list):
    h, w = latents_shape[-2:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    
    total_pattern = torch.zeros_like(theta, dtype=torch.complex64)
    
    for cfg in config_list:
        k = cfg['k']
        r_s = cfg['r_start'] / (h / 2)
        r_e = cfg['r_end'] / (h / 2)
        scale = cfg.get('scale', 1.0) 
        
        mask = (r >= r_s) & (r < r_e)
        mask = mask.float()
        
        # 频域加权
        total_pattern += mask * torch.exp(1j * k * theta) * scale
        
    return total_pattern.unsqueeze(0).unsqueeze(0)

def apply_spatial_window(spatial_pattern, device):
    """
    [关键修改] 实现 Vignette (暗角) 窗口
    中心保留 1.0，四角平滑过渡到 0.0，彻底消除吉布斯效应带来的四角异色。
    """
    b, c, h, w = spatial_pattern.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing='ij')
    r = torch.sqrt(x**2 + y**2) # 范围 0 ~ 1.414 (corners)
    
    # 定义渐变区间
    # 0.0 ~ 0.85: 权重 1.0 (保留水印)
    # 0.85 ~ 1.0: 权重 Cosine Decay -> 0
    # > 1.0 (四角): 权重 0 (强制抹平)
    
    fade_start = 0.85
    fade_end = 1.0
    
    # 1. 基础全1
    mask = torch.ones_like(r)
    
    # 2. 计算 Fade 区域 (0.85 < r < 1.0)
    # 使用余弦平滑过渡: 0.5 * (1 + cos(pi * x)) 从 1 变到 0
    fade_region_mask = (r > fade_start) & (r < fade_end)
    # 归一化到 0~1
    norm_r = (r[fade_region_mask] - fade_start) / (fade_end - fade_start)
    decay_val = 0.5 * (1 + torch.cos(3.1415926 * norm_r))
    mask[fade_region_mask] = decay_val
    
    # 3. 截断区域 (r >= 1.0) -> 置0
    mask[r >= fade_end] = 0.0
    
    return spatial_pattern * mask.unsqueeze(0).unsqueeze(0)

def eval_ring_score(latents, k_target, r_start, r_end, target_channel):
    device = latents.device
    # 检测时同样应用 Window，保证匹配滤波的一致性
    template_freq_raw = get_hybrid_pattern(latents.shape, device, [{'k': k_target, 'r_start': r_start, 'r_end': r_end}])
    template_spatial = torch.fft.ifft2(torch.fft.ifftshift(template_freq_raw))
    template_spatial_windowed = apply_spatial_window(template_spatial, device)
    template_freq_final = torch.fft.fftshift(torch.fft.fft2(template_spatial_windowed))
    template_freq_final = template_freq_final.squeeze(0).squeeze(0)
    
    fft_latents = torch.fft.fft2(latents.float())
    fft_latents = torch.fft.fftshift(fft_latents) 
    img_freq = fft_latents[:, target_channel, :, :] 
    
    correlation = img_freq * torch.conj(template_freq_final)
    score = torch.abs(torch.sum(correlation, dim=(-1, -2)))
    score = score / (latents.shape[-1] * latents.shape[-2])
    return score.item()

async def diff_watermark(idx, edit_timestep, seed, output_folder_timestep, gt_patch, current_prompt_embeddings, null_embeddings, current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device, pipe, args):
    set_random_seed(seed)
    final_result = {}
    output_dir_path = f'{output_folder_timestep}/img{idx}'
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)
    output_txt_file = f'{output_dir_path}/distance.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_txt_file, 'a') as txt_f: txt_f.write(f'{current_time}\n')

    xT_no_w_latent = pipe.get_random_latents()
    xt_no_w_latent = pipe.backward_diffusion(latents=xT_no_w_latent, text_embeddings=current_prompt_embeddings, text_embeddings_null=null_embeddings, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, reverse_process=False, start_timestep=0, end_timestep=args.num_inference_steps-edit_timestep)
    xt_w_latent = copy.deepcopy(xt_no_w_latent)
    
    target_user_id = 177 

    if args.w_pattern == 'spiral':
        watermark_config, debug_info = get_watermark_config_256(target_user_id)
        
        # 整体强度保持 1.0，依靠 Ring0 的 1.2 内部权重来增强
        w_strength = 1.0 
        
        with open(output_txt_file, 'a') as txt_f:
            txt_f.write(f"=== 256-ID Final Scheme (ID={target_user_id}) ===\n")
            txt_f.write(f"Config: {watermark_config}\nBase Strength: {w_strength}\n")
            
        freq_patch_complex = get_hybrid_pattern(xt_w_latent.shape, device, watermark_config)
        
        freq_unshifted = torch.fft.ifftshift(freq_patch_complex)
        spatial_patch_complex = torch.fft.ifft2(freq_unshifted)
        spatial_patch_real = spatial_patch_complex.real
        
        # 归一化
        spatial_patch_real = spatial_patch_real / (spatial_patch_real.std() + 1e-6) * w_strength
        
        # Soft Clipping 削峰
        spatial_patch_real = torch.clamp(spatial_patch_real, min=-3.0, max=3.0)
        
        # 应用新的 Vignette Window
        spatial_patch_real = apply_spatial_window(spatial_patch_real, device)
        
        w_noise = torch.zeros_like(xt_w_latent)
        w_noise[:, args.w_channel, :, :] = spatial_patch_real
        xt_w_latent = xt_w_latent + w_noise
        
        gt_patch = freq_patch_complex
        mask_visual = freq_patch_complex[0, 0].abs().cpu().numpy()
        plt.imsave(f'{output_dir_path}/vis_mask.png', mask_visual, cmap='gray')
        watermarking_mask_eval = mask_visual 
    else:
        watermarking_mask = get_watermarking_mask(init_latents_w=xt_w_latent, w_mask_shape=args.w_mask_shape, w_radius=args.w_radius, w_channel=args.w_channel, device=device)
        watermarking_mask_eval = watermarking_mask.clone()
        xt_w_latent = inject_watermark(xt_w_latent, watermarking_mask, gt_patch, w_injection=args.w_injection)

    x0_no_w_latent = pipe.backward_diffusion(latents=xt_no_w_latent, text_embeddings=current_prompt_embeddings, text_embeddings_null=null_embeddings, guidance_scale=1.0, num_inference_steps=args.num_inference_steps, reverse_process=False, start_timestep=args.num_inference_steps-edit_timestep, end_timestep=-1)
    x0_no_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_no_w_latent))[0]
    store_pil_image(x0_no_w_img, f'{output_dir_path}/x0_no_w.png')

    x0_w_latent = pipe.backward_diffusion(latents=xt_w_latent, text_embeddings=current_prompt_embeddings, text_embeddings_null=null_embeddings, guidance_scale=1.0, num_inference_steps=args.num_inference_steps, reverse_process=False, start_timestep=args.num_inference_steps-edit_timestep, end_timestep=-1)
    x0_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_w_latent))[0]
    store_pil_image(x0_w_img, f'{output_dir_path}/x0_w.png')

    arr_no_w = np.array(x0_no_w_img).astype(float)
    arr_w = np.array(x0_w_img).astype(float)
    diff = np.abs(arr_w - arr_no_w) * 10 
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(f'{output_dir_path}/vis_residual_diff.png')

    if args.reference_model is not None:
        sims = measure_similarity([x0_no_w_img, x0_w_img], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        w_no_sim, w_sim = sims[0].item(), sims[1].item()
    else:
        w_no_sim, w_sim = 0, 0
    with open(f'{output_dir_path}/distance.txt', 'a') as f:
        f.write('='*15 + f'\nclip score: no_w={w_no_sim}, w={w_sim}\n')
    final_result['clip_scores_no_w'] = w_no_sim
    final_result['clip_scores_w'] = w_sim

    no_w_img_class = one_image(clear_img=x0_no_w_img, label='no_w')
    w_img_class = one_image(clear_img=x0_w_img, label='w')
    img_class_list = [no_w_img_class, w_img_class]

    attackers = {
        'none': image_distortion_none,
        'jpeg': image_distortion_jpeg,
        'gaussianblur': image_distortion_gaussianblur,
        'gaussianstd': image_distortion_gaussianstd,
        'colorjitter': image_distortion_colorjitter,
        'randomdrop': image_distortion_randomdrop,
        'saltandpepper': image_distortion_saltandpepper,
        'resizerestore': image_distortion_resizerestore,
        'vaebmshj': image_distortion_vae1,
        'vaecheng': image_distortion_vae2,
        'diff': image_distortion_diff,
        'medianblur': image_distortion_medianblur,
        'diffpure': image_distortion_diffpure,
    }

    for attacker_name, attacker in attackers.items():
        if attacker_name == 'diffpure':
            attacker(img_class_list, 42, current_prompt_embeddings, null_embeddings, pipe, args)
        else:
            attacker(img_class_list, 42, args)
        
        for each_img_class in img_class_list:
            store_imgs(each_img_class, output_dir_path)

        for each_img_class in img_class_list:
            preprocessed_img = transform_img(getattr(each_img_class, attacker_name)).unsqueeze(0).to(null_embeddings.dtype).to(device)
            preprocessed_img = TF.gaussian_blur(preprocessed_img, kernel_size=21, sigma=4.0)

            image_latents = pipe.get_image_latents(preprocessed_img, sample=False)
            reversed_latents = pipe.forward_diffusion(latents=image_latents, text_embeddings=null_embeddings, guidance_scale=1.0, num_inference_steps=args.num_inference_steps, start_timestep=0, end_timestep=edit_timestep)
            
            with open(output_txt_file, 'a') as txt_f:
                txt_f.write(f'*'*50 + f'\n{attacker_name} (Argmax+AlignedBlur)\n')
                if args.w_pattern == 'spiral':
                    _, expected_info = get_watermark_config_256(target_user_id)
                    detected_id = 0
                    for info in expected_info:
                        r_s, r_e = info['range']
                        pool = info['pool']
                        gt_state = info['state']
                        scores = []
                        for k_cand in pool:
                            s = eval_ring_score(reversed_latents, k_cand, r_s, r_e, args.w_channel)
                            scores.append(s)
                        det_state = np.argmax(scores)
                        match_mark = "[MATCH]" if det_state == gt_state else "[FAIL]"
                        txt_f.write(f"R{info['ring_idx']}: GT={gt_state} Det={det_state} {match_mark} | Scores: {[f'{x:.4f}' for x in scores]}\n")
                        detected_id += det_state * (4 ** info['ring_idx'])
                    txt_f.write(f"Final: GT={target_user_id}, Det={detected_id}\n")
                    acc = 1.0 if detected_id == target_user_id else 0.0
                    eval_results = {'spiral_score': acc, 'accuracy': acc}
                else:
                    eval_results = eval_watermark_single(reversed_latents, watermarking_mask_eval, gt_patch, args.w_measurement, args.w_channel)
            record_results(output_txt_file, eval_results, f'{args.w_measurement}_{each_img_class.label}')
            final_result[f'{each_img_class.label}_metrics_{attacker_name}'] = get_metrics(eval_results, args.w_measurement)
    return final_result

async def main(args, edit_t, pipe, scheduler, dataset, prompt_key, ref_model, ref_clip_preprocess, ref_tokenizer):
    edit_timestep = int(edit_t * args.num_inference_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipe.to(device)
    tester_prompt = '' 
    null_embeddings = pipe.get_text_embedding(tester_prompt)
    final_result_list = {}
    prefixes = ['no_w_metrics', 'w_metrics']
    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter','randomdrop', 'saltandpepper', 'resizerestore','vaebmshj', 'vaecheng', 'diff', 'medianblur', 'diffpure']
    final_result_list.update({f'{prefix}_{suffix}': [] for prefix in prefixes for suffix in suffixes})
    clip_suffixes = ['no_w', 'w']
    final_result_list.update({f'clip_scores_{suffix}': [] for suffix in clip_suffixes})
    output_folder_timestep = f'output/{args.run_name}/timestep{edit_timestep}'
    if not os.path.exists(output_folder_timestep): os.makedirs(output_folder_timestep)
    with open(os.path.join('output', args.run_name, 'config.log'), 'a') as txt_f: txt_f.write(f'{args}\n')
    
    print(f"[Info] Watermark Channel: {args.w_channel} (Make sure this is 2 or 3 for best quality)")
    
    gt_patch = torch.zeros(1) if args.w_pattern == 'spiral' else get_watermarking_pattern(pipe, args, device)
    total_tasks = []
    for i in tqdm(range(args.start, args.end)):
        current_prompt = dataset[i][prompt_key]
        current_prompt_embeddings = pipe.get_text_embedding(current_prompt)
        seed = i + args.w_seed
        task = asyncio.create_task(diff_watermark(i, edit_timestep, seed, output_folder_timestep, gt_patch, current_prompt_embeddings, null_embeddings, current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device, pipe, args))
        total_tasks.append([task])
        if len(total_tasks) % 5 == 0:
            for t in itertools.chain.from_iterable(total_tasks): await t
            for st in total_tasks:
                res = st[0].result()
                for k, v in res.items(): final_result_list[k].append(v)
            total_tasks = []
    if len(total_tasks) != 0:
        for t in itertools.chain.from_iterable(total_tasks): await t
        for st in total_tasks:
            res = st[0].result()
            for k, v in res.items(): final_result_list[k].append(v)
    attacker_names = suffixes
    for key, value in final_result_list.items():
        if 'no_w' in key or 'clip' in key or 'clear_img' in key: continue
        for attacker_name in attacker_names:
            if attacker_name in key:
                base_result_list = final_result_list[f'no_w_metrics_{attacker_name}']
        get_roc_auc(base_result_list, final_result_list[key], args.w_measurement, key, f'{output_folder_timestep}/overall_scores.txt')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(args.model_id, scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)
    ref_model = None
    ref_clip_preprocess = None
    ref_tokenizer = None
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
    dataset, prompt_key = get_dataset(args)
    initialize_attackers(args, device)
    for edit_t in args.edit_time_list:
        asyncio.run(main(args, edit_t=edit_t, pipe=pipe, scheduler=scheduler, dataset=dataset, prompt_key=prompt_key, ref_model=ref_model, ref_clip_preprocess=ref_clip_preprocess, ref_tokenizer=ref_tokenizer))
    print('finished.')