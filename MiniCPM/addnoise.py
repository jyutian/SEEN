import argparse
import sys
import numpy as np
import soundfile as sf
import os
from scipy.signal import resample_poly

SEED = 42
np.random.seed(SEED)

TARGET_SR = 16000

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


def generate_gaussian_noise(length, snr_db, clean):
 
    clean_power = np.mean(clean ** 2)
    
    target_noise_power = clean_power / (10 ** (snr_db / 10))
   
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--noise_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--snrs", type=str, default="10"
                       )
    parser.add_argument("--include_gaussian", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')

    args = parser.parse_args()

    snr_values = [int(s.strip()) for s in args.snrs.split(",")]

    if not os.path.isdir(args.clean_dir):
        print(f"[Error] clean_dir not exist: {args.clean_dir}")
        sys.exit(1)

    if not os.path.isdir(args.noise_root):
        print(f"[Error] noise_root not exist: {args.noise_root}")
        sys.exit(1)

    clean_files = [f for f in os.listdir(args.clean_dir) if f.endswith(".wav")]
    noise_files = [f for f in os.listdir(args.noise_root) if f.endswith(".wav")]

    if not clean_files:
        print("[Error] clean_dir doesn't have wav 文件")
        sys.exit(1)

    if not noise_files:
        print("[Error] noise_root doesn't have wav 文件")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for noise_file in noise_files:
        noise_path = os.path.join(args.noise_root, noise_file)
        noise_name = os.path.splitext(noise_file)[0]

        print(f"\n=== noise type: {noise_name} ===")

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

               
                if args.skip_existing and os.path.exists(noisy_output_path) and os.path.exists(raw_output_path):
                    print(f"Skip {clean_file}, file already exists")
                    continue

                add_noise_with_snr(
                    clean_path,
                    noise_audio,
                    noisy_output_path,
                    raw_output_path,
                    snr=snr_v,
                    is_gaussian=False 
                )


    
    print("\n=== Process Gauss ===")
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

           
            if args.skip_existing and os.path.exists(noisy_output_path) and os.path.exists(raw_output_path):
                print(f"Skip {clean_file}, file already exists")
                continue

            add_noise_with_snr(
                clean_path,
                None, 
                noisy_output_path,
                raw_output_path,
                snr=snr_v,
                is_gaussian=True
                )

    print("\n All noise types have been done")
