import argparse
import os
from datetime import datetime
from pathlib import Path
import jiwer
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import re
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

class OutputInspector:
    def __init__(self, target_layer):
        self.layer_outputs = []
        self.handle = target_layer.activation_fn.register_forward_hook(self._hook_fn)
        self.qkv_cache = {}

    def _hook_fn(self, module, input, output):
        self.layer_outputs.append(output.detach().cpu())

    def get_outputs(self):
        return torch.cat(self.layer_outputs, dim=0), self.qkv_cache

    def release(self):
        self.handle.remove()
   
 
class AudioDenoisePipeline:
    def __init__(self, model, processor, target_layer):
        self.model = model
        self.processor = processor
        self.target_layer = target_layer
        
    def _run(self,audio_path):
        USE_AUDIO_IN_VIDEO = False
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": "You are an automatic speech recognition system.Transcribe the audio exactly as spoken.Do not summarize, do not translate, do not add or remove any words."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path}
                ],
            },
        ]
        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                           use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pattern = r"assistant\s*(.*)"
        match = re.search(pattern, text[0], flags=re.S)
        if match:
            answer = match.group(1).strip()
        else:
            answer = text.strip()
        print(f"Transcription: {answer}")
        return answer

    def generate_with_denoise(self, audio_path,V_noise_list):
        res = self._run(audio_path)
        handles = []
        for layer, Vn in zip(self.target_layer, V_noise_list):
            handles.append(Deactivate(layer, Vn))

        # forward
        res_new = self._run(audio_path)

        raw_acts, deacts = [], []
        E_raw, E_denoise = [], []
        F_raw, F_denoise = [], []

        for h in handles:
            # ---- activation ----
            raw_acts.append(h.raw_hidden.mean(dim=0, keepdim=True))
            deacts.append(h.denoised_hidden.mean(dim=0, keepdim=True))
            E_raw.append(h.E_raw)
            E_denoise.append(h.E_de)
            F_raw.append(h.F_raw)
            F_denoise.append(h.F_de)

            h.release()

        deact = torch.cat(deacts, dim=0)
        raw_act = torch.cat(raw_acts, dim=0)
        print(f"raw_act.shape:{raw_act.shape}, deact.shape:{deact.shape}")
        return res,res_new,raw_act,deact,E_raw,F_raw,E_denoise,F_denoise


class Deactivate:
    def __init__(self, target_layer, V_noise):
        self.target_layer = target_layer
        self.V_noise = V_noise
        self.outputHandle = target_layer.activation_fn.register_forward_hook(self.denoise_hook)
        self.denoised_hidden = None
        self.raw_hidden = None
        self.E_raw = None
        self.F_raw = None
        self.E_de = None
        self.F_de = None
    def computer_ef(self,acts,Vn):
        coeffs = torch.matmul(acts, Vn)   # (T,k)
        proj_energy = (coeffs**2).sum(dim=1)  # (T)
        total_energy = (acts**2).sum(dim=1)  # (T)
        
        e = proj_energy.mean().item()
        f = (proj_energy / (total_energy + 1e-12)).mean().item()
        return e,f
    def denoise_hook(self, module, input, output, alpha=1):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        V_noise = self.V_noise
        # print((V_noise.T @ V_noise))

        # print(f"[Hook] hidden={hidden.shape}, Vn={V_noise.shape}")
        orig_dtype = hidden.dtype
        device = hidden.device

        # hidden: (T, D)
        x = hidden.to(torch.float32)
        self.raw_hidden = x.detach().cpu()
        Vn = V_noise.to(device, dtype=torch.float32)
        e,f = self.computer_ef(x,Vn)
        self.E_raw = e
        self.F_raw = f 
        proj_noise = x @ Vn @ Vn.T
        x = x - alpha * proj_noise
        e,f = self.computer_ef(x,Vn)
        # print(f"Denoise Hook: Projected Energy E={E:.6f}, Frame Ratio F={F:.6f}")
        self.E_de = e
        self.F_de = f
        hidden = x.to(orig_dtype)
        self.denoised_hidden = hidden.detach().cpu()
        # 恢复 tuple 输出结构
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            return hidden

    def release(self):
        self.outputHandle.remove()


def read_csv(csv_path,test_samples=100):
    """
    从metadata.csv中读取音频路径,提取前num_samples条音频的中间层激活。
    """
    df = pd.read_csv(csv_path)
    
    audio_test_path = []
    audio_noise_path = []
    audio_noise_test_path = []
    answer = []
    for i, row in tqdm(df.iterrows(), total=min(test_samples, len(df))):
        if i >= test_samples:
            break
        audio_test_path.append(row["audio_id"])
        base_name = os.path.basename(row["audio_id"])
        audio_noise_path.append(f'../datasets/{dataset}/noise/{noise}/snr_{snr}/raw/{base_name}')
        audio_noise_test_path.append(f'../datasets/{dataset}/noise/{noise}/snr_{snr}/audio_noise/{base_name}')
        answer.append(row["text"])
    return audio_test_path, audio_noise_test_path,answer
def ASR(hyp_list, ref_list):
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

    refs = [transform(x) for x in ref_list]
    hyps = [transform(x) for x in hyp_list]

    wer_value = jiwer.wer(refs, hyps)
    return wer_value
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example for dataset, noise, method, test_len")

    # 添加命令行参数
    parser.add_argument('--datasets', type=str, default='librispeech',choices=['librispeech'], help='Dataset name')
    parser.add_argument('--noise', type=str, default='gauss', help='Noise type')
   
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--test_len', type=int, default=100, help='test-len')
    parser.add_argument('--SNR', type=int, default=0, choices=[-10,-5,0,5,10,20,30],help='SNR')
    # 解析命令行参数
    
    args = parser.parse_args()
    dataset = args.datasets
    noise = args.noise
    gpu_id = args.gpu
    test_len = args.test_len
    snr = args.SNR
    # 使用 args 参数
    print("datasets:",args.datasets)
    print("Noise:", args.noise)
   
    print("gpu_id", args.gpu)
    print("test_len", args.test_len)
    print("SNR", args.SNR)
    local_model = 'model'
    csv_path = f'../datasets/{dataset}/metadata.csv'
    method = 'SEES'
    os.makedirs(f"log_{method}/{noise}", exist_ok=True)
    log_path = f'log_{method}/{noise}/{dataset}.txt'
    os.makedirs(f"res_{method}/{noise}/{dataset}", exist_ok=True)
    os.makedirs(f"sim_{method}/{noise}/{dataset}", exist_ok=True)
    res_path = f"res_{method}/{noise}/{dataset}/{snr}_{noise.replace('/','_')}.csv"
    sim_path = f"sim_{method}/{noise}/{dataset}/{snr}_{noise.replace('/','_')}.csv"
    model_path = 'model/Qwen/Qwen2___5-Omni-7B'
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=None,trust_remote_code=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.disable_talker()
    device = torch.device(f"cuda:{gpu_id}")  # 你想用的 GPU
    model.to(device)
    model.eval()

    res_new = []
    res_old = []
    res = []
    res_noise = []

    audio_paths, audio_noise_paths,answer = read_csv(csv_path,test_samples=test_len)
    pipeline = AudioDenoisePipeline(model, processor, model.thinker.audio_tower.layers[23:])

    V_list = torch.load(f'v_list_{noise}_0.1_0.pt')
    # E,F = pipeline.getef(audio_test_paths,test_texts,v_list)
    # 在测试集上评估
    sim1_list, sim2_list, sim3_list, sim4_list, sim5_list, sim6_list = [], [], [], [], [], []
    E,F,E_noise,F_noise,E_new ,F_new,E_old,F_old = [],[],[],[],[],[],[],[]
    for i in range(test_len):
        print(f"-----------Process On:{i + 1}/{test_len}条音频-----------")
        print(f"{audio_paths[i]}")
        print(f"{audio_noise_paths[i]}")
        result,result_old,act,deact,e,e_old,f,f_old = pipeline.generate_with_denoise(audio_paths[i],V_list)
        result_noise,result_new,act_noise,deact_noise,e_noise,e_new,f_noise,f_new = pipeline.generate_with_denoise(audio_noise_paths[i],V_list)
        E.append(e)
        F.append(f)
        E_noise.append(e_noise)
        F_noise.append(f_noise)
        E_new.append(e_new)
        F_new.append(f_new)
        E_old.append(e_old)
        F_old.append(f_old)
        
        res_new.append(result_new)
        res_old.append(result_old)
        res.append(result)
        res_noise.append(result_noise)
        sim1 = torch.nn.functional.cosine_similarity(act, act_noise, dim=1).mean()
        sim2 = torch.nn.functional.cosine_similarity(act, deact_noise, dim=1).mean()
        sim3 = torch.nn.functional.cosine_similarity(deact, act_noise, dim=1).mean()
        sim4 = torch.nn.functional.cosine_similarity(act, deact, dim=1).mean()
        sim5 = torch.nn.functional.cosine_similarity(deact, deact_noise, dim=1).mean()
        sim6 = torch.nn.functional.cosine_similarity(act_noise, deact_noise, dim=1).mean()
        sim1_list.append(sim1.float().item())
        sim2_list.append(sim2.float().item())
        sim3_list.append(sim3.float().item())
        sim4_list.append(sim4.float().item())
        sim5_list.append(sim5.float().item())
        sim6_list.append(sim6.float().item())
    columns = ["act-act_noise", "act-deact_noise", "deact-act_noise", 
           "act-deact", "deact-deact_noise", "act_noise-deact_noise"]

    # 创建 DataFrame
    df = pd.DataFrame({
        columns[0]: sim1_list,
        columns[1]: sim2_list,
        columns[2]: sim3_list,
        columns[3]: sim4_list,
        columns[4]: sim5_list,
        columns[5]: sim6_list,
    })

    # 写入 CSV 文件
    df.to_csv(sim_path, index=False, encoding='utf-8')

    os.makedirs(f'{method}_{dataset}/{noise}/yuan', exist_ok=True)
    os.makedirs(f'{method}_{dataset}/{noise}/noise', exist_ok=True)
    os.makedirs(f'{method}_{dataset}/{noise}/new', exist_ok=True)
    os.makedirs(f'{method}_{dataset}/{noise}/old', exist_ok=True)
    
    df = pd.DataFrame(E, columns=[f'Layer_{i}_E' for i in range(len(E[0]))])
    df['row_mean_E'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/yuan/E.csv", index=False, header=True)
    df = pd.DataFrame(F, columns=[f'Layer_{i}_F' for i in range(len(F[0]))])
    df['row_mean_F'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/yuan/F.csv", index=False, header=True)
    df = pd.DataFrame(E_noise, columns=[f'Layer_{i}_E' for i in range(len(E_noise[0]))])
    df['row_mean_E'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/noise/E_{snr}.csv", index=False, header=True)
    df = pd.DataFrame(F_noise, columns=[f'Layer_{i}_F' for i in range(len(F_noise[0]))])
    df['row_mean_F'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/noise/F_{snr}.csv", index=False, header=True)
    df = pd.DataFrame(E_new, columns=[f'Layer_{i}_E' for i in range(len(E_new[0]))])
    df['row_mean_E'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/new/E_{snr}.csv", index=False, header=True)
    df = pd.DataFrame(F_new, columns=[f'Layer_{i}_F' for i in range(len(F_new[0]))])
    df['row_mean_F'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/new/F_{snr}.csv", index=False, header=True)
    df = pd.DataFrame(E_old, columns=[f'Layer_{i}_E' for i in range(len(E_old[0]))])
    df['row_mean_E'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/old/E_{snr}.csv", index=False, header=True)
    df = pd.DataFrame(F_old, columns=[f'Layer_{i}_F' for i in range(len(F_old[0]))])
    df['row_mean_F'] = df.mean(axis=1)
    df.to_csv(f"{method}_{dataset}/{noise}/old/F_{snr}.csv", index=False, header=True)

    r_2_vs_1 = ASR(res, answer)
    r_3_vs_1 = ASR(res_old, answer)
    r_4_vs_1 = ASR(res_noise, answer)
    r_5_vs_1 = ASR(res_new, answer)

    r_3_vs_2 = ASR(res, res_old)
    r_4_vs_2 = ASR(res, res_noise)
    r_5_vs_2 = ASR(res, res_new)
    print("===== ASR结果汇总 =====")
    print(f"正常 vs 正确答案 WER: {r_2_vs_1:.2%}")
    print(f"正常抑制后 vs 正确答案 WER: {r_3_vs_1:.2%}")
    print(f"加噪 vs 正确答案 WER: {r_4_vs_1:.2%}")
    print(f"加噪抑制后 vs 正确答案 WER: {r_5_vs_1:.2%}")
    print(f"正常抑制后 vs 正常 WER: {r_3_vs_2:.2%}")
    print(f"加噪 vs 正常 WER: {r_4_vs_2:.2%}")
    print(f"加噪抑制后 vs 正常 WER: {r_5_vs_2:.2%}")

    row = {
    "dataset": dataset,
    "noise": noise,
    "time": datetime.now().isoformat(timespec="seconds"),
    "正常 vs 正确答案 WER": f"{r_2_vs_1:.2%}",
    "正常抑制后 vs 正确答案 WER": f"{r_3_vs_1:.2%}",
    "加噪 vs 正确答案 WER": f"{r_4_vs_1:.2%}",
    "加噪抑制后 vs 正确答案 WER": f"{r_5_vs_1:.2%}",
    "正常抑制后 vs 正常 WER": f"{r_3_vs_2:.2%}",
    "加噪 vs 正常 WER": f"{r_4_vs_2:.2%}",
    "加噪抑制后 vs 正常 WER": f"{r_5_vs_2:.2%}",
    }

    df = pd.DataFrame([row])

    # 判断是否首次写入
    write_header = not os.path.exists(res_path)

    df.to_csv(
        res_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8"
    )
        

