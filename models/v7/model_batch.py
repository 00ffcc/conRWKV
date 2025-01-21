########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py, 增加state
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from typing import List
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch._C._jit_set_autocast_mode(False)


# DTYPE = torch.bfloat16
DTYPE = torch.half # better

########################################################################################################
# RWKV TimeMix
########################################################################################################

class RWKV_Tmix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.dtype = DTYPE

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd
        
        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, args.D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(args.D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, args.D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(args.D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, args.D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(args.D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, args.D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(args.D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

    def forward(self, x, v_first, state:List[List[torch.Tensor]]):
        B, T, C = x.shape
        H = self.n_head
        x_prev = state[self.layer_id][0]
        xx = torch.cat([x_prev.view(B, 1, C), x[:, :-1, :]], dim=1) - x
        state[self.layer_id][0] = x[:, -1, :]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x, state[self.layer_id][1] = self.RWKV7_OP(r, w, k, v, -kk, kk*a, state[self.layer_id][1])
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        
        return x, v_first
    
    def RWKV7_OP(self, r, w, k, v, a, b, state):
        B, T, C = r.shape
        H = C // self.head_size
        N = self.head_size
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)
        return out.view(B, T, C).to(dtype=self.dtype), state
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x, state:List[List[torch.Tensor]]):
        B, T, C = x.shape
        x_prev = state[self.layer_id][2]
        xx = torch.cat([x_prev.view(B, 1, C), x[:, :-1, :]], dim=1) - x
        state[self.layer_id][2] = x[:, -1, :]

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block
########################################################################################################

class Block(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first, state:List[List[torch.Tensor]]):

        xx, v_first = self.att(self.ln1(x), v_first, state)
        x = x + xx
        x = x + self.ffn(self.ln2(x), state)

        return x, v_first

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.ln0 = nn.LayerNorm(args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    @staticmethod
    def from_pretrained(model_path):
        z = torch.load(model_path, map_location='cpu')
        z['ln0.weight'] = z['blocks.0.ln0.weight']
        z['ln0.bias'] = z['blocks.0.ln0.bias']
        args = types.SimpleNamespace()
        args.n_head, args.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = args.n_head * args.head_size
        args.n_layer = max(int(k.split('.')[1]) for k in z.keys() if 'blocks.' in k) + 1
        args.vocab_size = z['emb.weight'].shape[0]
        args.D_DECAY_LORA = z['blocks.1.att.w1'].shape[1]
        args.D_AAA_LORA = z['blocks.1.att.a1'].shape[1]
        args.D_MV_LORA = z['blocks.1.att.v1'].shape[1]
        args.D_GATE_LORA = z['blocks.1.att.g1'].shape[1]
        model = RWKV(args)
        model.load_state_dict(z, strict=False)
        model = model.to('cuda').to(dtype=DTYPE)
        return model

    def forward(self, idx, state):

        x = self.emb(idx)
        x = self.ln0(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first, state)

        x = self.ln_out(x)
        x = self.head(x)

        return x
    def empty_state(self, batch_size):
        return [
                [
                    torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, device='cuda'),
                    torch.zeros(batch_size, self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size, dtype=torch.float32, device='cuda'),
                    torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, device='cuda'),
                ] 
                for _ in range(self.args.n_layer)]
    

########################################################################################################
# RWKV Inference
########################################################################################################

if __name__ == '__main__':
    with torch.no_grad():

        import sys
        sys.path.append(r"..\..")
        from tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
        tokenizer=RWKVWorldTokenizer(vocab_file=r"..\..\tokenizer\rwkv_vocab_v20230424.txt")
        MODEL_NAME = r"..\..\weights\v7-0.1b.pth"

        model = RWKV.from_pretrained(MODEL_NAME)
        model.eval()

        ########################################################################################################

        prompt = "The Eiffel tower is in the city of"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        print(f'\nInput:\n{input_ids}')
        state = model.empty_state(1)
        out = model.forward(input_ids[:, :4], state)
        print(state)
        out = model.forward(input_ids[:, 4:], state)
        print(f'\nOutput:\n{out}')

        # logits of the last token => prediction for the next token    
        out = out[0, -1]
        
        probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

        print(f'\n{prompt}')

        _, indices = torch.topk(probs, 10) # print top-10 possibilities
        for i in range(len(indices)):
            token_id = indices[i].item()
            token = tokenizer.decode([token_id])
            token_prob = probs[token_id].item()
            print(token, f'[probability {token_prob:.2%}]')

