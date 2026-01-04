import argparse
import sys
import numpy as np
import soundfile as sf
import os
from scipy.signal import resample_poly

# =====================================================
# 全局随机种子（保证可复现）
# =====================================================
SEED = 42
np.random.seed(SEED)

TARGET_SR = 16000


# =====================================================
# 工具函数
# =====================================================
def resample_to_16k(audio, orig_sr):
    if orig_sr == TARGET_SR:
        return audio
    return resample_poly(audio, TARGET_SR, orig_sr)


def adjust_noise_to_snr(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
    return noise * scale


def loop_or_cut_noise(noise, target_length):
    if len(noise) >= target_length:
        return noise[:target_length]
    repeat = target_length // len(noise) + 1
    return np.tile(noise, repeat)[:target_length]


# 生成高斯噪声并根据目标SNR调整
def generate_gaussian_noise(length, snr_db, clean):
    # 计算目标音频的功率
    clean_power = np.mean(clean ** 2)
    # 根据SNR计算噪声功率
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    # 生成标准差为sqrt(噪声功率)的高斯噪声
    std = np.sqrt(target_noise_power)
    noise = np.random.normal(0, std, length)
    return noise


def add_noise_with_snr(clean_path, noise_array, noisy_output_path, raw_output_path, snr, is_gaussian=False):
    clean, sr = sf.read(clean_path)
    if clean.ndim > 1:
        clean = clean[:, 0]

    clean = resample_to_16k(clean, sr)

    if is_gaussian:
        noise = generate_gaussian_noise(len(clean), snr, clean)
    else:
        noise = loop_or_cut_noise(noise_array, len(clean))

    noise_scaled = adjust_noise_to_snr(clean, noise, snr)
    noisy = clean + noise_scaled

    sf.write(raw_output_path, noise_scaled, TARGET_SR)
    sf.write(noisy_output_path, noisy, TARGET_SR)


# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--noise_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--snrs", type=str, default="10",
                        help="例如: 0,10,20,30")
    parser.add_argument("--include_gaussian", action='store_true', 
                        help="是否包括高斯噪声")
    parser.add_argument("--skip_existing", action='store_true', 
                        help="如果目标文件已存在，则跳过生成")

    args = parser.parse_args()

    snr_values = [int(s.strip()) for s in args.snrs.split(",")]

    if not os.path.isdir(args.clean_dir):
        print(f"[错误] clean_dir 不存在: {args.clean_dir}")
        sys.exit(1)

    if not os.path.isdir(args.noise_root):
        print(f"[错误] noise_root 不存在: {args.noise_root}")
        sys.exit(1)

    clean_files = [f for f in os.listdir(args.clean_dir) if f.endswith(".wav")]
    noise_files = [f for f in os.listdir(args.noise_root) if f.endswith(".wav")]

    if not clean_files:
        print("[错误] clean_dir 中没有 wav 文件")
        sys.exit(1)

    if not noise_files:
        print("[错误] noise_root 中没有 wav 文件")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # =================================================
    # 处理真实噪声 wav 和高斯噪声
    # =================================================
    for noise_file in noise_files:
        noise_path = os.path.join(args.noise_root, noise_file)
        noise_name = os.path.splitext(noise_file)[0]

        print(f"\n=== 噪声类型: {noise_name} ===")

        noise_audio, noise_sr = sf.read(noise_path)
        if noise_audio.ndim > 1:
            noise_audio = noise_audio[:, 0]

        noise_audio = resample_to_16k(noise_audio, noise_sr)

        for snr_v in snr_values:
            snr_dir = os.path.join(
                args.output_dir,
                noise_name,
                f"snr_{snr_v}"
            )

            raw_dir = os.path.join(snr_dir, "raw")
            noisy_dir = os.path.join(snr_dir, "audio_noise")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(noisy_dir, exist_ok=True)

            for clean_file in clean_files:
                clean_path = os.path.join(args.clean_dir, clean_file)

                raw_output_path = os.path.join(raw_dir, clean_file)
                noisy_output_path = os.path.join(noisy_dir, clean_file)

                # 如果文件已存在，跳过处理
                if args.skip_existing and os.path.exists(noisy_output_path) and os.path.exists(raw_output_path):
                    print(f"跳过 {clean_file}，文件已存在")
                    continue

                add_noise_with_snr(
                    clean_path,
                    noise_audio,
                    noisy_output_path,
                    raw_output_path,
                    snr=snr_v,
                    is_gaussian=False  # 这里仍然使用现有噪声文件
                )

    # 处理高斯噪声
    
    print("\n=== 处理高斯噪声 ===")
    for snr_v in snr_values:
        snr_dir = os.path.join(
            args.output_dir,
            "gauss",
            f"snr_{snr_v}"
        )

        raw_dir = os.path.join(snr_dir, "raw")
        noisy_dir = os.path.join(snr_dir, "audio_noise")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(noisy_dir, exist_ok=True)

        for clean_file in clean_files:
            clean_path = os.path.join(args.clean_dir, clean_file)

            raw_output_path = os.path.join(raw_dir, clean_file)
            noisy_output_path = os.path.join(noisy_dir, clean_file)

            # 如果文件已存在，跳过处理
            if args.skip_existing and os.path.exists(noisy_output_path) and os.path.exists(raw_output_path):
                print(f"跳过 {clean_file}，文件已存在")
                continue

            add_noise_with_snr(
                clean_path,
                None,  # 高斯噪声不需要噪声文件
                noisy_output_path,
                raw_output_path,
                snr=snr_v,
                is_gaussian=True  # 指定使用高斯噪声
                )

    print("\n✅ 所有噪声添加完成（随机种子已固定，采样率 16kHz）")
