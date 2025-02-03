########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# 笑死，刚准备改就发现fla加了rwkv7的layer和model
# 1.21号晚上发现cu_seqlens有问题，想修一下，结果22号Zhiyuan Li就修了，太强了orz
########################################################################################################

import torch, types
import torch.nn as nn
from typing import List, Optional, Union
from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
from torch.nn import functional as F
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

    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor, 
                state: List[List[torch.Tensor]],
                cu_seqlens: torch.LongTensor
                ):
        B, T, C = x.shape
        H = self.n_head
        N = self.head_size

        xx = torch.cat([torch.empty(1, 1, C, device=x.device, dtype=x.dtype), x[:, :-1, :]], dim=1)
        
        N_prefill = cu_seqlens.shape[-1] - 1
        xx[0, cu_seqlens[:-1], :] = state[self.layer_id][0][:N_prefill]
        xx[0, cu_seqlens[-1]:, :] = state[self.layer_id][0][N_prefill:]

        state[self.layer_id][0][:N_prefill] = x[0, cu_seqlens[1:]-1, :]
        state[self.layer_id][0][N_prefill:] = x[0, cu_seqlens[-1]:, :]

        xx = xx - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -torch.exp(-F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5) # soft-clamp to (-inf, -0.5)
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



        r.resize_((B, T, H, N))
        w.resize_((B, T, H, N))
        k.resize_((B, T, H, N))
        v.resize_((B, T, H, N))
        kk.resize_((B, T, H, N))
        a.resize_((B, T, H, N))

        N_prefill = cu_seqlens.shape[-1] - 1
        if N_prefill > 0:
            # prefilling
            r_  =  r[:, :cu_seqlens[-1], :, :]
            w_  =  w[:, :cu_seqlens[-1], :, :]
            k_  =  k[:, :cu_seqlens[-1], :, :]
            v_  =  v[:, :cu_seqlens[-1], :, :]
            kk_ = kk[:, :cu_seqlens[-1], :, :]
            a_  =  a[:, :cu_seqlens[-1], :, :]

            x1, state[self.layer_id][1][:N_prefill] = chunk_rwkv7(
                r=r_,
                log_w=w_,
                k=k_,
                v=v_,
                a=-kk_,
                b=kk_ * a_,
                scale=1.,
                initial_state=state[self.layer_id][1][:N_prefill],
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                head_first=False
            )
        else:
            x1 = torch.empty(B, 0, H, N, device=x.device, dtype=x.dtype)
            
        if cu_seqlens[-1] < T:
            # decoding
            r_  =  r[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            w_  =  w[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            k_  =  k[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            v_  =  v[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            kk_ = kk[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            a_  =  a[:, cu_seqlens[-1]:, :, :].view(-1, 1, H, N)
            
            x2, state[self.layer_id][1][N_prefill:] = fused_recurrent_rwkv7(
                r=r_,
                log_w=w_,
                k=k_,
                v=v_,
                a=-kk_,
                b=kk_ * a_,
                scale=1.,
                initial_state=state[self.layer_id][1][N_prefill:],
                output_final_state=True,
                cu_seqlens=None,
                head_first=False
            )
            x2 = x2.view(B, T-cu_seqlens[-1], H, N)
        else:
            x2 = torch.empty(B, 0, H, N, device=x.device, dtype=x.dtype)
        
        x = torch.cat([x1, x2], dim=1)

        # fla的state最后2维和官方实现(https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py)是反的，要转置一下
        
        x.resize_((B * T, C))
        x = self.ln_x(x).view(B, T, C)

        x = x + ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).view(B, T, C)
        x = self.output(x * g)

        v_first.resize_((B, T, C))
        return x, v_first
    
    
########################################################################################################
# RWKV ChannelMix
########################################################################################################

class RWKV_CMix_x070(torch.nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, 
                x: torch.Tensor, 
                state: List[List[torch.Tensor]], 
                cu_seqlens: torch.LongTensor
                ):
        B, T, C = x.shape

        xx = torch.cat([torch.empty(1, 1, C, device=x.device, dtype=x.dtype), x[:, :-1, :]], dim=1)
        N_prefill = cu_seqlens.shape[-1] - 1
        xx[0, cu_seqlens[:-1], :] = state[self.layer_id][2][:N_prefill]
        xx[0, cu_seqlens[-1]:, :] = state[self.layer_id][2][N_prefill:]

        state[self.layer_id][2][:N_prefill] = x[0, cu_seqlens[1:]-1, :]
        state[self.layer_id][2][N_prefill:] = x[0, cu_seqlens[-1]:, :]

        xx = xx - x
        

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
        
    def forward(self, x, v_first, state:List[List[torch.Tensor]], cu_seqlens):

        xx, v_first = self.att(self.ln1(x), v_first, state, cu_seqlens)
        x = x + xx
        x = x + self.ffn(self.ln2(x), state, cu_seqlens)

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
    def from_pretrained(model_path, device):
        z = torch.load(model_path, mmap=True, weights_only=True)
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
        model = model.to(device).to(dtype=DTYPE)
        model.device = device
        return model

    def forward(self, 
                idx: Union[torch.LongTensor, List[torch.LongTensor]], 
                state: List[List[List[torch.Tensor]]], 
                ):
        '''
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N_of_prefill+1]` used for variable-length training,
            consistent with the FlashAttention API.  
            the start index of each segment.
            idx[0, cu_seqlens[-1]:] are the decoding tokens.
        '''

        assert isinstance(idx, list), f"idx must be a list of LongTensors"
        # sort idx by length in descending order
        idx, state, index = zip(*sorted(zip(idx, state, range(len(idx))), key=lambda x: x[0].shape[-1], reverse=True))
        cu_seqlens = torch.cat([torch.LongTensor([0]), torch.cumsum(torch.LongTensor([i.shape[-1] for i in idx if i.shape[-1]>1]), dim=0)], dim=0).to(self.device)

        idx = torch.cat(idx, dim=-1)
        state = [[torch.cat([state[j][k] for state in state], dim=0) for k in range(3)] for j in range(self.args.n_layer)]

        reverse_index = [index.index(i) for i in range(len(index))]

        assert idx.shape[0] == 1, "batch size must be 1"
        assert state[0][0].shape[0] == (cu_seqlens.shape[0] - 1) + (idx.shape[-1] - cu_seqlens[-1]), f"state shape error {state[0][0].shape[0]} {cu_seqlens.shape[0]-1} {idx.shape[-1] - cu_seqlens[-1]}"

        x = self.emb(idx)
        x = self.ln0(x)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first, state, cu_seqlens)

        x = self.ln_out(x)
        x = self.head(x)

        state = [[[state[j][k][i:i+1] for k in range(3)] for j in range(self.args.n_layer)] for i in reverse_index]
        
        x = torch.cat([x[0, cu_seqlens[1:]-1], x[0, cu_seqlens[-1]:]], dim=0)
        x = torch.stack([x[i] for i in reverse_index], dim=0)
        
        return x, state
    def empty_state(self, batch_size=1, device=None):
        if device is None:
            device = self.device
        return [
                [
                    torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, device=device),
                    torch.zeros(batch_size, self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size, dtype=torch.float32, device=device),
                    torch.zeros(batch_size, self.args.n_embd, dtype=DTYPE, device=device),
                ] 
                for _ in range(self.args.n_layer)]
    def move_state(self, state, device=None):
        if device is None:
            device = self.device
        return [[s.to(device) for s in layer] for layer in state]

########################################################################################################
# RWKV Inference
########################################################################################################

if __name__ == '__main__':
    with torch.no_grad():

        import sys
        sys.path.append(r"../..")
        from tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
        tokenizer=RWKVWorldTokenizer(vocab_file=r"../../tokenizer/rwkv_vocab_v20230424.txt")
        MODEL_NAME = r"../../weights/v7-0.1b.pth"

        model = RWKV.from_pretrained(MODEL_NAME, device='cuda')
        model.eval()

        ########################################################################################################

        prompt = "The Eiffel tower is in the city of"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        print(f'\nInput:\n{input_ids}')
        state = [model.empty_state(1) for _ in range(3)]
        outs, state = model.forward([input_ids[:, :4], input_ids[:, :6], input_ids[:, :5]], state)
        outs, state = model.forward([input_ids[:, 4:-1], input_ids[:, 6:-1], input_ids[:, 5:-1]], state)
        outs, state = model.forward([input_ids[:, -1:], input_ids[:, -1:], input_ids[:, -1:]], state)
        # out, state = model.forward([input_ids], state)
        # print(f'\nOutput:\n{out}')

        # logits of the last token => prediction for the next token    
        for i in range(outs.shape[0]):
            out = outs[i]
            
            probs = F.softmax(out.float(), dim=-1) # compute softmax in float (more accurate)

            print(f'\n{prompt}')

            _, indices = torch.topk(probs, 10) # print top-10 possibilities
            for i in range(len(indices)):
                token_id = indices[i].item()
                token = tokenizer.decode([token_id])
                token_prob = probs[token_id].item()
                print(token, f'[probability {token_prob:.2%}]')


        # input_ids = torch.zeros((1, 1024), dtype=torch.long, device='cuda')
        # state = model.empty_state(1)
        # import time
        # start_time = time.time()
        # model.forward([input_ids[:, :200]], state)
        # print(f"Time elapsed: {time.time() - start_time:.2f}s")
        # model.forward([input_ids[:, :800]], state)
        # print(f"Time elapsed: {time.time() - start_time:.2f}s")
        # # model.forward([input_ids[:, :500]], state)
        # # print(f"Time elapsed: {time.time() - start_time:.2f}s")