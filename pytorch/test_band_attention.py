import time
import torch
import band_attention

bs = 20
nh = 8
nt = 200
channel = 128
window = 4
n_iter = 100

for it in range(n_iter):
    device = "cuda:0"
    Q = torch.randn(bs, nh, nt, channel, dtype=torch.float32, device=device)
    K = torch.randn(bs, nh, nt, channel, dtype=torch.float32, device=device)
    V = torch.randn(bs, nh, nt, channel, dtype=torch.float32, device=device)
    attn = torch.zeros(bs, nh, nt, nt, dtype=torch.float32, device=device)
    X = torch.zeros(bs, nh, nt, channel, dtype=torch.float32, device=device)

    start_time_1 = time.time()
    standard_attn = Q @ K.transpose(-2, -1)
    standard_attn = standard_attn.softmax(dim=-1)
    standard_X = standard_attn @ V
    end_time_1 = time.time()

    start_time_2 = time.time()
    standard_attn = Q @ K.transpose(-2, -1)
    attn_mask = torch.zeros_like(attn)
    for i in range(nt):
        attn_mask[:, :, i, max(i-window, 0) : min(i+window+1, nt)] = 1
    attn_bias = torch.zeros_like(attn)
    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    standard_attn = standard_attn + attn_bias
    standard_attn = standard_attn.softmax(dim=-1)
    standard_X = standard_attn @ V
    end_time_2 = time.time()

    start_time_3 = time.time()
    band_attention.torch_launch_band_attention(X.reshape(-1), attn.reshape(-1), Q.reshape(-1), K.reshape(-1), V.reshape(-1), window, bs, nh, nt, channel)
    X = X.reshape(bs, nh, nt, channel)
    end_time_3 = time.time()

    print("time standard (full attention): ", end_time_1 - start_time_1)
    print("time standard (masked self attention): ", end_time_2- start_time_2)
    print("time CUDA accelerated band attention: ", end_time_3 - start_time_3)
    print()