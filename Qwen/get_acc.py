import argparse
import os
from datetime import datetime
from pathlib import Path
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
        
    def _run(self, audio_path, text):
        USE_AUDIO_IN_VIDEO = False
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.Answer the multiple-choice question by outputting ONLY one letter."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path}, {"type": "text", "text": text}
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

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        match = re.search(r'assistant\s+([A-Za-z])', text)
        return match.group(1)

    def generate_with_denoise(self, audio_path, text, V_noise_list):
        res = self._run(audio_path, text)
        handles = []
        for layer, Vn in zip(self.target_layer, V_noise_list):
            handles.append(Deactivate(layer, Vn))

        # forward
        res_new = self._run(audio_path, text)

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
       
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            return hidden

    def release(self):
        self.outputHandle.remove()


def read_csv(csv_path,test_samples=100):
   
    df = pd.read_csv(csv_path)
    
    audio_test_path = []
    audio_noise_path = []
    audio_noise_test_path = []
   
    text_test = []
    answer = []
    for i, row in tqdm(df.iterrows(), total=min(test_samples, len(df))):
        if i >= test_samples:
            break
        audio_test_path.append(row["audio_id"])
        base_name = os.path.basename(row["audio_id"])
        audio_noise_path.append(f'../datasets/{dataset}/noise/{noise}/snr_{snr}/raw/{base_name}')
        audio_noise_test_path.append(f'../datasets/{dataset}/noise/{noise}/snr_{snr}/audio_noise/{base_name}')
        text_test.append(row["question"]+row["choices"])
        answer.append(row["answer"])
    return audio_test_path, audio_noise_test_path,text_test, answer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example for dataset, noise, method, test_len")

    parser.add_argument('--datasets', type=str, choices=['music','sound','speech'], help='Dataset name')
    parser.add_argument('--noise', type=str, default='gauss', help='Noise type')
   
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--test_len', type=int, default=300, help='test-len')
    parser.add_argument('--SNR', type=int, default=0, choices=[-10,-5,0,5,10,20,30],help='SNR')
    
    args = parser.parse_args()
    dataset = args.datasets
    noise = args.noise
    gpu_id = args.gpu
    test_len = args.test_len
    snr = args.SNR
  
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
    model_path = './model'
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=None,
                                                                trust_remote_code=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.disable_talker()
    device = torch.device(f"cuda:{gpu_id}")  
    model.to(device)
    model.eval()

    res_new = []
    res_old = []
    res = []
    res_noise = []

    audio_paths, audio_noise_paths, texts,res_correct = read_csv(csv_path,test_samples=test_len)
    print(res_correct)
    
    pipeline = AudioDenoisePipeline(model, processor, model.thinker.audio_tower.layers[23:])

    V_list = torch.load(f'v_list_{noise}_0.1_0.pt')
    # E,F = pipeline.getef(audio_test_paths,test_texts,v_list)
   
    sim1_list, sim2_list, sim3_list, sim4_list, sim5_list, sim6_list = [], [], [], [], [], []
    E,F,E_noise,F_noise,E_new ,F_new,E_old,F_old = [],[],[],[],[],[],[],[]
    for i in range(test_len):
        print(f"-----------Process On:{i + 1}/{test_len}audio-----------")
        print(f"{audio_paths[i]}")
        print(f"{audio_noise_paths[i]}")
        result,result_old,act,deact,e,e_old,f,f_old = pipeline.generate_with_denoise(audio_paths[i], texts[i],V_list)
        result_noise,result_new,act_noise,deact_noise,e_noise,e_new,f_noise,f_new = pipeline.generate_with_denoise(audio_noise_paths[i], texts[i],V_list)
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

    
    df = pd.DataFrame({
        columns[0]: sim1_list,
        columns[1]: sim2_list,
        columns[2]: sim3_list,
        columns[3]: sim4_list,
        columns[4]: sim5_list,
        columns[5]: sim6_list,
    })

    
    df.to_csv(sim_path, index=False, encoding='utf-8')

    Path(res_path).parent.mkdir(parents=True, exist_ok=True) 
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(f"--------{datetime.now()}--------\n")

    df = pd.DataFrame({
        'answer': res_correct,
        'res': res,
        'res_old': res_old,
        'res_noise': res_noise,
        'res_new': res_new,
    })
    df.to_csv(res_path, index=False, encoding='utf-8')
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

    df = pd.read_csv(res_path)

   
    fixed = df[(df["res_noise"] != df["answer"]) & (df["res_new"] == df["answer"])]
    fixed_len = len(df[(df["res_noise"] != df["answer"])])
    
    broken = df[(df["res_noise"] == df["answer"]) & (df["res_new"] != df["answer"])]
    broken_len = len(df[(df["res_noise"] == df["answer"])])
    
    chang = df[(df["res_noise"] != df["res"]) & (df["res_new"] == df["res"])]
    chang_len = len(df[(df["res_noise"] != df["res"])])
   
    yi = df[(df["res_noise"] == df["res"]) & (df["res_new"] != df["res"])]
    yi_len = len(df[(df["res_noise"] == df["res"])])
    
    print(fixed_len, broken_len)
    y_true =(df["answer"] == df["res"]).astype(int).values

  
    y_pred_noise = (df["res_noise"] == df["res"]).astype(int).values


    y_pred_new = (df["res_new"] == df["res"]).astype(int).values

    def report_preds(y_true, y_pred, name):
        F1_micro =  f1_score(y_true, y_pred, average='micro')
        F1_macro = f1_score(y_true, y_pred, average='macro')
        print(f"\n=== Report for: {name} ===")
        print("Samples:", len(y_true))
        print("F1 (micro)   :", F1_micro)
        print("F1 (macro)   :", F1_macro)
        # print("F1 (weighted):", f1_score(y_true, y_pred, average='weighted'))
        print("\nPer-class classification report:\n")
        print(classification_report(y_true, y_pred, digits=4))
        return F1_micro,F1_macro

    
    F1_micro_noise,F1_macro_noise = report_preds(y_true, y_pred_noise, "res_noise vs res (noisy prediction)")
    F1_micro_new,F1_macro_new = report_preds(y_true, y_pred_new,   "res_new   vs res (suppressed prediction)")
    def radio(correct, pred):
        correct = np.array(correct)
        pred = np.array(pred)
        ratio = np.mean(correct == pred)
        return ratio


    r_2_vs_1 = radio(res, res_correct)
    r_3_vs_1 = radio(res_old, res_correct)
    r_4_vs_1 = radio(res_noise, res_correct)
    r_5_vs_1 = radio(res_new, res_correct)

    r_3_vs_2 = radio(res, res_old)
    r_4_vs_2 = radio(res, res_noise)
    r_5_vs_2 = radio(res, res_new)
    
    args_dict = vars(args)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--------{dataset}:result{datetime.now()}--------\n")
        for k, v in args_dict.items():
            f.write(f"{k} = {v}\n")
          
        f.write(f"Clean vs Ground Truth agreement rate: {r_2_vs_1:.2%}\n")
        f.write(f"Denoised (clean) vs Ground Truth agreement rate: {r_3_vs_1:.2%}\n")
        f.write(f"Noisy vs Ground Truth agreement rate: {r_4_vs_1:.2%}\n")
        f.write(f"Noisy + Denoised vs Ground Truth agreement rate: {r_5_vs_1:.2%}\n")
        
        f.write(f"Denoised (clean) vs Clean agreement rate: {r_3_vs_2:.2%}\n")
        f.write(f"Noisy vs Clean agreement rate: {r_4_vs_2:.2%}\n")
        f.write(f"Noisy + Denoised vs Clean agreement rate: {r_5_vs_2:.2%}\n")
        
        f.write(
            f"Incorrect under noise but corrected after denoising: "
            f"{len(fixed)}/{fixed_len} = {len(fixed)/fixed_len:.2%}\n"
        )
        f.write(
            f"Correct under noise but incorrect after denoising: "
            f"{len(broken)}/{broken_len} = {len(broken)/broken_len:.2%}\n"
        )
        f.write(
            f"Abnormal under noise but normalized after denoising: "
            f"{len(chang)}/{chang_len} = {len(chang)/chang_len:.2%}\n"
        )
        f.write(
            f"Normal under noise but abnormal after denoising: "
            f"{len(yi)}/{yi_len} = {len(yi)/yi_len:.2%}\n"
        )

        f.write(f"F1_micro_noise:{F1_micro_noise:.4%}\n")
        f.write(f"F1_macro_noise:{F1_macro_noise:.4%}\n")
        f.write(f"F1_micro_new:{F1_micro_new:.4%}\n")
        f.write(f"F1_macro_new:{F1_macro_new:.4%}\n")



