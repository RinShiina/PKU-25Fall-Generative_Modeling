import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
import os
import json

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
import seaborn as sns  # 用于画热力图，如果报错没这个库，请 pip install seaborn
import torch.fft
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==============================================================================
# [TEST CONFIG] 64-ID Scheme (Same as t2i_64ID.py)
# ==============================================================================

def get_watermark_config_64(user_id):
    user_id = int(user_id) % 64
    rings_spatial = [(10, 14), (16, 20), (22, 26)]
    # 候选频率池 (与你提供的 t2i_64ID.py 保持一致)
    candidate_pools = [
        [7, 11, 15, 19],  # R0
        [9, 13, 17, 21],  # R1
        [11, 15, 19, 23]  # R2
    ]
    config_list = []
    debug_info = []
    temp_id = user_id
    for i in range(3):
        state = temp_id % 4
        temp_id = temp_id // 4
        r_s, r_e = rings_spatial[i]
        pool = candidate_pools[i]
        target_k = pool[state]
        config_list.append({'k': target_k, 'r_start': r_s, 'r_end': r_e})
        debug_info.append({'ring_idx': i, 'range': (r_s, r_e), 'state': state, 'pool': pool, 'target_k': target_k})
    return config_list, debug_info


def get_hybrid_pattern(latents_shape, device, config_list):
    h, w = latents_shape[-2:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device),
                          indexing='ij')
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    total_pattern = torch.zeros_like(theta, dtype=torch.complex64)
    for cfg in config_list:
        k = cfg['k']
        r_s = cfg['r_start'] / (h / 2)
        r_e = cfg['r_end'] / (h / 2)
        mask = (r >= r_s) & (r < r_e)
        mask = mask.float()
        total_pattern += mask * torch.exp(1j * k * theta)
    return total_pattern.unsqueeze(0).unsqueeze(0)


def apply_spatial_window(spatial_pattern, device):
    b, c, h, w = spatial_pattern.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device),
                          indexing='ij')
    r = torch.sqrt(x ** 2 + y ** 2)
    spatial_mask = torch.clamp(1 - (torch.abs(r - 0.55) - 0.3) / 0.15, 0, 1)
    spatial_mask[r > 1.0] = 0
    return spatial_pattern * spatial_mask.unsqueeze(0).unsqueeze(0)


def eval_ring_score(latents, k_target, r_start, r_end, target_channel):
    device = latents.device
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


async def diff_watermark(idx, edit_timestep, seed, output_folder_timestep, gt_patch, current_prompt_embeddings,
                         null_embeddings, current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device, pipe,
                         args):
    set_random_seed(seed)
    final_result = {}

    # 动态分配 ID: 0-1000 循环覆盖 0-63
    target_user_id = idx % 64

    output_dir_path = f'{output_folder_timestep}/img{idx}_id{target_user_id}'
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)
    output_txt_file = f'{output_dir_path}/distance.txt'

    # 仅在第一次运行时打印详细信息，避免 log 爆炸
    if idx == args.start:
        with open(output_txt_file, 'a') as txt_f: txt_f.write(f'Processing ID={target_user_id}\n')

    xT_no_w_latent = pipe.get_random_latents()
    xt_no_w_latent = pipe.backward_diffusion(latents=xT_no_w_latent, text_embeddings=current_prompt_embeddings,
                                             text_embeddings_null=null_embeddings, guidance_scale=args.guidance_scale,
                                             num_inference_steps=args.num_inference_steps, reverse_process=False,
                                             start_timestep=0, end_timestep=args.num_inference_steps - edit_timestep)
    xt_w_latent = copy.deepcopy(xt_no_w_latent)

    if args.w_pattern == 'spiral':
        watermark_config, debug_info = get_watermark_config_64(target_user_id)
        w_strength = 1.5
        freq_patch_complex = get_hybrid_pattern(xt_w_latent.shape, device, watermark_config)
        freq_unshifted = torch.fft.ifftshift(freq_patch_complex)
        spatial_patch_complex = torch.fft.ifft2(freq_unshifted)
        spatial_patch_real = spatial_patch_complex.real
        spatial_patch_real = spatial_patch_real / (spatial_patch_real.std() + 1e-6) * w_strength
        spatial_patch_real = apply_spatial_window(spatial_patch_real, device)
        w_noise = torch.zeros_like(xt_w_latent)
        w_noise[:, args.w_channel, :, :] = spatial_patch_real
        xt_w_latent = xt_w_latent + w_noise
        gt_patch = freq_patch_complex
        mask_visual = freq_patch_complex[0, 0].abs().cpu().numpy()
        plt.imsave(f'{output_dir_path}/vis_mask.png', mask_visual, cmap='gray')
        watermarking_mask_eval = mask_visual
    else:
        # Fallback for old patterns
        watermarking_mask = get_watermarking_mask(init_latents_w=xt_w_latent, w_mask_shape=args.w_mask_shape,
                                                  w_radius=args.w_radius, w_channel=args.w_channel, device=device)
        watermarking_mask_eval = watermarking_mask.clone()
        xt_w_latent = inject_watermark(xt_w_latent, watermarking_mask, gt_patch, w_injection=args.w_injection)

    x0_no_w_latent = pipe.backward_diffusion(latents=xt_no_w_latent, text_embeddings=current_prompt_embeddings,
                                             text_embeddings_null=null_embeddings, guidance_scale=1.0,
                                             num_inference_steps=args.num_inference_steps, reverse_process=False,
                                             start_timestep=args.num_inference_steps - edit_timestep, end_timestep=-1)
    x0_no_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_no_w_latent))[0]
    store_pil_image(x0_no_w_img, f'{output_dir_path}/x0_no_w.png')

    x0_w_latent = pipe.backward_diffusion(latents=xt_w_latent, text_embeddings=current_prompt_embeddings,
                                          text_embeddings_null=null_embeddings, guidance_scale=1.0,
                                          num_inference_steps=args.num_inference_steps, reverse_process=False,
                                          start_timestep=args.num_inference_steps - edit_timestep, end_timestep=-1)
    x0_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_w_latent))[0]
    store_pil_image(x0_w_img, f'{output_dir_path}/x0_w.png')

    # Metric: CLIP Score (Optional, simplified)
    w_no_sim, w_sim = 0, 0
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

    # Record ID info for aggregation
    final_result['target_id'] = target_user_id

    for attacker_name, attacker in attackers.items():
        if attacker_name == 'diffpure':
            attacker(img_class_list, 42, current_prompt_embeddings, null_embeddings, pipe, args)
        else:
            attacker(img_class_list, 42, args)

        # Save distorted images only for the first few to save space/time
        if idx < 5:
            for each_img_class in img_class_list:
                store_imgs(each_img_class, output_dir_path)

        for each_img_class in img_class_list:
            if each_img_class.label == 'no_w': continue  # Skip detection on no_w for speed

            preprocessed_img = transform_img(getattr(each_img_class, attacker_name)).unsqueeze(0).to(
                null_embeddings.dtype).to(device)
            # Aligned Blur
            preprocessed_img = TF.gaussian_blur(preprocessed_img, kernel_size=21, sigma=4.0)

            image_latents = pipe.get_image_latents(preprocessed_img, sample=False)
            reversed_latents = pipe.forward_diffusion(latents=image_latents, text_embeddings=null_embeddings,
                                                      guidance_scale=1.0, num_inference_steps=args.num_inference_steps,
                                                      start_timestep=0, end_timestep=edit_timestep)

            if args.w_pattern == 'spiral':
                _, expected_info = get_watermark_config_64(target_user_id)
                detected_id = 0
                for info in expected_info:
                    r_s, r_e = info['range']
                    pool = info['pool']
                    scores = []
                    for k_cand in pool:
                        s = eval_ring_score(reversed_latents, k_cand, r_s, r_e, args.w_channel)
                        scores.append(s)
                    det_state = np.argmax(scores)
                    detected_id += det_state * (4 ** info['ring_idx'])

                acc = 1.0 if detected_id == target_user_id else 0.0
                # Save specifically for aggregation
                final_result[f'acc_{attacker_name}'] = acc
            else:
                # Fallback
                pass

    return final_result


def visualize_statistics(stats, output_dir):
    # Prepare data
    attacker_names = list(next(iter(stats.values())).keys())
    attacker_names.sort()
    ids = sorted(list(stats.keys()))

    # 1. Heatmap: ID vs Attacker
    heatmap_data = np.zeros((len(attacker_names), len(ids)))
    for i, atk in enumerate(attacker_names):
        for j, uid in enumerate(ids):
            data = stats[uid][atk]
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            heatmap_data[i, j] = acc

    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, xticklabels=ids, yticklabels=attacker_names, annot=False, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Accuracy Heatmap: ID vs Attacker")
    plt.xlabel("User ID")
    plt.ylabel("Attacker")
    plt.savefig(os.path.join(output_dir, "id_accuracy_heatmap.png"))
    plt.close()

    # 2. Bar Chart: Average Accuracy per Attacker
    avg_accs = []
    for atk in attacker_names:
        total_correct = sum(stats[uid][atk]['correct'] for uid in ids)
        total_count = sum(stats[uid][atk]['total'] for uid in ids)
        avg_accs.append(total_correct / total_count if total_count > 0 else 0)

    plt.figure(figsize=(12, 6))
    plt.bar(attacker_names, avg_accs, color='skyblue')
    plt.xticks(rotation=45)
    plt.title("Average Accuracy per Attacker (Global)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    for i, v in enumerate(avg_accs):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_average_accuracy.png"))
    plt.close()

    # Save raw json
    with open(os.path.join(output_dir, "full_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=4)


async def main(args, edit_t, pipe, scheduler, dataset, prompt_key, ref_model, ref_clip_preprocess, ref_tokenizer):
    edit_timestep = int(edit_t * args.num_inference_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipe.to(device)
    tester_prompt = ''
    null_embeddings = pipe.get_text_embedding(tester_prompt)

    output_folder_timestep = f'output/{args.run_name}/test_stats'
    if not os.path.exists(output_folder_timestep): os.makedirs(output_folder_timestep)

    print(f"[Info] Running Comprehensive Test for 64 IDs...")
    print(f"[Info] Range: {args.start} to {args.end}")

    gt_patch = torch.zeros(1)  # Not used for spiral

    # Initialize Statistics: {id: {attacker: {correct: 0, total: 0}}}
    stats = {i: {} for i in range(64)}
    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter', 'randomdrop', 'saltandpepper',
                'resizerestore', 'vaebmshj', 'vaecheng', 'diff', 'medianblur', 'diffpure']
    for i in range(64):
        for atk in suffixes:
            stats[i][atk] = {'correct': 0, 'total': 0}

    total_tasks = []

    # Run loop
    for i in tqdm(range(args.start, args.end)):
        current_prompt = dataset[i][prompt_key]
        current_prompt_embeddings = pipe.get_text_embedding(current_prompt)
        seed = i + args.w_seed

        task = asyncio.create_task(
            diff_watermark(i, edit_timestep, seed, output_folder_timestep, gt_patch, current_prompt_embeddings,
                           null_embeddings, current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device, pipe,
                           args))
        total_tasks.append(task)

        # Batch size control for concurrency
        if len(total_tasks) >= 5:
            results = await asyncio.gather(*total_tasks)
            for res in results:
                uid = res['target_id']
                for atk in suffixes:
                    acc = res.get(f'acc_{atk}', 0)
                    stats[uid][atk]['total'] += 1
                    stats[uid][atk]['correct'] += int(acc)
            total_tasks = []

    # Cleanup remaining tasks
    if len(total_tasks) > 0:
        results = await asyncio.gather(*total_tasks)
        for res in results:
            uid = res['target_id']
            for atk in suffixes:
                acc = res.get(f'acc_{atk}', 0)
                stats[uid][atk]['total'] += 1
                stats[uid][atk]['correct'] += int(acc)

    # Visualize
    visualize_statistics(stats, output_folder_timestep)
    print(f"[Info] Statistics saved to {output_folder_timestep}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(args.model_id, scheduler=scheduler,
                                                             torch_dtype=torch.float16, use_safetensors=True)
    pipe.set_progress_bar_config(disable=True)

    # Load dataset
    dataset, prompt_key = get_dataset(args)

    # Initialize attackers
    initialize_attackers(args, device)

    for edit_t in args.edit_time_list:
        asyncio.run(main(args, edit_t=edit_t, pipe=pipe, scheduler=scheduler, dataset=dataset, prompt_key=prompt_key,
                         ref_model=None, ref_clip_preprocess=None, ref_tokenizer=None))
    print('finished.')