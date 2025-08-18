import torch
from torch import nn
from models import VGGTransformer


vocab_size = 100
n_layers = 2
d_model = 32
d_ff = 64
h = 4
p_dropout = 0.1
in_features = 40
batch_size = 2
seq_len_enc = 15
seq_len_dec = 27

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==== Tạo dữ liệu đầu vào ====
src = torch.randn(batch_size, seq_len_enc, in_features)                   # encoder input
tgt = torch.randint(0, vocab_size, (batch_size, seq_len_dec))            # decoder input

# ==== Tạo mask ====
src_mask = torch.ones(batch_size, seq_len_enc)           # [B, 1, M, T]
tgt_mask = torch.ones(batch_size, 1, seq_len_dec, seq_len_dec)           # [B, 1, M, M]


config = {
    'input_dim': in_features,
    'in_features': 1280,
    'n_enc_layers': n_layers,
    'd_model': d_model,
    'ff_size': d_ff,
    'h': h,
    'p_dropout': p_dropout,
    'n_dec_layers': n_layers,
    'model_name': 'VGGTransformer'
}
model = VGGTransformer(config, vocab_size).to(device)
src = src.to(device)
tgt = tgt.to(device)
src_mask = src_mask.to(device)
tgt_mask = tgt_mask.to(device)
enc_out, dec_out, enc_input_lengths  = model(src, tgt, src_mask, tgt_mask)
print("Encoder Output Shape:", enc_out.shape)  # [B, T, vocab_size]
print("Decoder Output Shape:", dec_out.shape)  # [B, M, vocab_size]
print("Encoder Input Lengths Shape:", enc_input_lengths.shape)  # [B]



