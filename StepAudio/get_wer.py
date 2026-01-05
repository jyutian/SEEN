import os
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm

from stepaudio2 import StepAudio2
from token2wav import Token2wav   
import argparse
import os
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm
import jiwer
class OutputInspector:
    def __init__(self, target_layer):
        self.layer_outputs = []
        self.handle = target_layer.mlp[1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.layer_outputs.append(output.detach().cpu())

    def get_outputs(self):
        return torch.cat(self.layer_outputs, dim=0)

    def release(self):
        self.handle.remove()
 
class AudioDenoisePipeline:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
    def _run(self,audio_path):
        messages = [
            {"role": "system", "content": "You are a speech recognition system. Please record the content of the speech you hear.Do not add or remove any words."},
            {"role": "human", "content": [{"type": "audio", "audio": f"{audio_path}"}]},
            {"role": "assistant", "content": None}
        ]
            
        tokens, text, _ = model(messages, max_new_tokens=256)
        print(text)
        return text

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
            raw_acts.append(h.raw_hidden.mean(dim=(0,1)).unsqueeze(0))
            deacts.append(h.denoised_hidden.mean(dim=(0,1)).unsqueeze(0))
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
        self.outputHandle = target_layer.mlp[1].register_forward_hook(self.denoise_hook)
        self.denoised_hidden = None
        self.raw_hidden = None
        self.E_raw = None
        self.F_raw = None
        self.E_de = None
        self.F_de = None
    def computer_ef(self,acts,Vn):
        acts =acts.to(torch.float32)
   
        coeffs = torch.matmul(acts, Vn)   # (B,T,k)
        proj_energy = (coeffs**2).mean(dim=0).sum(dim=1)  # (T)
        total_energy = (acts**2).mean(dim=0).sum(dim=1)  # (T)
        
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
        self.denoised_hidden = x.detach().cpu()
        hidden = x.to(orig_dtype)
        
      
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

   
    parser.add_argument('--datasets', type=str, default='librispeech',choices=['librispeech'], help='Dataset name')
    parser.add_argument('--noise', type=str, default='gauss', help='Noise type')
   
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--test_len', type=int, default=100, help='test-len')
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
    
    csv_path = f'../datasets/{dataset}/metadata.csv'
    method = 'SEES'
    os.makedirs(f"log_{method}/{noise}", exist_ok=True)
    log_path = f'log_{method}/{noise}/{dataset}.txt'
    os.makedirs(f"res_{method}/{noise}/{dataset}", exist_ok=True)
    os.makedirs(f"sim_{method}/{noise}/{dataset}", exist_ok=True)
    res_path = f"res_{method}/{noise}/{dataset}/{snr}_{noise.replace('/','_')}.csv"
    sim_path = f"sim_{method}/{noise}/{dataset}/{snr}_{noise.replace('/','_')}.csv"
    model = StepAudio2('model')
    token2wav = Token2wav('model/token2wav')
    target_layer = model.llm.encoder.blocks[27:]
    res_new ,res_old, res,res_noise= [],[],[],[]
    res_new = []
    res_old = []
    res = []
    res_noise = []

    audio_paths, audio_noise_paths,answer = read_csv(csv_path,test_samples=test_len)
    pipeline = AudioDenoisePipeline(model,target_layer)

    V_list = torch.load(f'v_list_{noise}_0.1_0.pt')
    # E,F = pipeline.getef(audio_test_paths,test_texts,v_list)
    
    sim1_list, sim2_list, sim3_list, sim4_list, sim5_list, sim6_list = [], [], [], [], [], []
    E,F,E_noise,F_noise,E_new ,F_new,E_old,F_old = [],[],[],[],[],[],[],[]
    for i in range(test_len):
        print(f"-----------Process On:{i + 1}/{test_len}audio-----------")
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

    
    df = pd.DataFrame({
        columns[0]: sim1_list,
        columns[1]: sim2_list,
        columns[2]: sim3_list,
        columns[3]: sim4_list,
        columns[4]: sim5_list,
        columns[5]: sim6_list,
    })

    
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
    
    row = {
    "dataset": dataset,
    "noise": noise,
    "time": datetime.now().isoformat(timespec="seconds"),

    "Clean vs Ground Truth WER": f"{r_2_vs_1:.2%}",
    "Denoised (Clean) vs Ground Truth WER": f"{r_3_vs_1:.2%}",
    "Noisy vs Ground Truth WER": f"{r_4_vs_1:.2%}",
    "Noisy + Denoised vs Ground Truth WER": f"{r_5_vs_1:.2%}",

    "Denoised (Clean) vs Clean WER": f"{r_3_vs_2:.2%}",
    "Noisy vs Clean WER": f"{r_4_vs_2:.2%}",
    "Noisy + Denoised vs Clean WER": f"{r_5_vs_2:.2%}",
}


    df = pd.DataFrame([row])

  
    write_header = not os.path.exists(res_path)

    df.to_csv(
        res_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8"
    )
        

