import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Bio import SeqIO
from tangermeme.plot import plot_logo
from layer import create

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded

def pad_sequence(seq, target_len=2000):
    real_len = len(seq)
    if real_len >= target_len:
        return seq[:target_len], target_len
    return seq + 'N' * (target_len - real_len), real_len

def custom_saturation_mutagenesis(model, X, device='cuda'):
    model.eval()
    seq_length = X.shape[2]
    X = X.to(device)
    with torch.no_grad():
        y0 = model(X)
    y = torch.zeros((X.shape[0], seq_length, 4), device=device)
    for i in range(seq_length):
        for j in range(4):
            X_mut = X.clone()
            X_mut[:, :, i] = 0
            X_mut[:, j, i] = 1
            with torch.no_grad():
                y_mut = model(X_mut)
            y[:, i, j] = y_mut - y0
    return y.cpu()

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['enhancer_score']

def save_ism_to_meme(ism_tensor, motif_name, out_path, real_len=None):
    ism_array = ism_tensor.squeeze(0).cpu().numpy()
    if real_len is not None:
        ism_array = ism_array[:real_len, :]
    ism_array = np.maximum(ism_array, 0)
    row_sums = ism_array.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-6
    pwm = ism_array / row_sums
    with open(out_path, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {motif_name}\n\n")
        f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} nsites= 20 E= 0\n")
        for row in pwm:
            f.write(" {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row))
    print(f"[✓] MEME motif saved to {out_path}")

def process_fasta_sequence(seq_name, sequence, model, device, output_dir):
    sequence, real_len = pad_sequence(sequence, target_len=2000)
    seq_ohe = one_hot_encode(sequence)
    seq_tensor = torch.FloatTensor(seq_ohe).unsqueeze(0).permute(0, 2, 1).to(device)
    wrapped_model = ModelWrapper(model).to(device)
    with torch.no_grad():
        ism = custom_saturation_mutagenesis(wrapped_model, seq_tensor, device=device)
    ism_valid = ism[0, :real_len, :].T
    plt.figure(figsize=(max(10, real_len // 100), 3))
    ax = plt.subplot(111)
    plot_logo(ism_valid, ax=ax)
    plt.xlabel("Genomic Position")
    plt.ylabel("Prediction Δ")
    plt.title(f"ISM: {seq_name} (first {real_len}bp)")
    out_img = os.path.join(output_dir, f'ism_logo_{seq_name}.png')
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.show()
    out_meme = os.path.join(output_dir, f'{seq_name}.meme')
    save_ism_to_meme(ism, motif_name=seq_name, out_path=out_meme, real_len=real_len)

# ==== 主程序入口 ====
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, required=True, help='FASTA文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    parser.add_argument('--outdir', type=str, default='./results/plots', help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model = create(
        channel1=512,
        channel2=384,
        channel3=128,
        channel4=200,
        channel5=200,
        embed_dim=128
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for record in SeqIO.parse(args.fasta, "fasta"):
        seq_name = record.id
        sequence = str(record.seq)
        print(f"[INFO] Processing {seq_name} ({len(sequence)}bp)")
        process_fasta_sequence(seq_name, sequence, model, device, args.outdir)
