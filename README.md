<h1 align="center">conRWKV</h1>

<div align="center">
    本项目旨在提供一个高并发的RWKV云端推理引擎，以方便后续基于RWKV的应用。

### feature

- 支持continuous batching，确保高吞吐量。
- prefill使用[fla(flash linear attention)](https://github.com/fla-org/flash-linear-attention/)，使用tensor core加速prefill，降低TTFT(Time to First Token)。
- 兼容OpenAI API。
- 支持RWKV v7。
- 极简安装部署

### install & deploy

```bash
pip install git+https://github.com/00ffcc/conRWKV
conRWKV --model /path/to/pth
```

**注意：**为了防止triton在处理变长数据时反复重新编译kernel，conRWKV中使用的[fla](https://github.com/00ffcc/flash-linear-attention)与[官方版本](https://github.com/fla-org/flash-linear-attention/)略有不同，安装前需要删除原有的fla或新开一个虚拟环境。

### benchmark

这里展示了与[RWKV-Runner](https://github.com/josStorer/RWKV-Runner)的比较。由于服务器上没有vulkan环境，暂时无法测试[Ai00](https://github.com/Ai00-X/ai00_server)，有测试的方法请戳我。

| RPS  | Num prompts | Engine      | Successful requests | Median E2E Latency | Median TTFT | Median ITL |
| ---- | ----------- | ----------- | ------------------- | ------------------ | ----------- | ---------- |
| 2    | 1200        | conRWKV     | **1200**            | 7750.72            | **115.76**  | 56.33      |
| 2    | 1200        | RWKV-Runner | **189**[^1]         | 3746.53            | **2176.03** | 16.29      |
| 4    | 1200        | conRWKV     | **1200**            | 10469.67           | **145.69**  | 77.45      |
| 4    | 1200        | RWKV-Runner | **119**[^1]         | 4484.30            | **1633.34** | 15.78      |

[^1]:不知道为什么RWKV-Runner只能完成很少一部分request，有知道的话请戳我

**测试环境：** RTX4090D x1, fp16

**测试命令：**

```bash
conRWKV --model /path/to/pth
python -m conRWKV.benchmark.benchmark --model rwkv --backend conRWKV --request-rate 2 --num-prompts 1200
python -m conRWKV.benchmark.benchmark --model rwkv --backend conRWKV --request-rate 4 --num-prompts 1200
```

```bash
python ./backend-python/main.py
curl http://127.0.0.1:8000/switch-model -X POST -H "Content-Type: application/json" -d '{"model":"/path/to/pth","strategy":"cuda fp16","deploy":"true"}'
python -m conRWKV.benchmark.benchmark --model rwkv --backend RWKV-Runner --request-rate 2 --num-prompts 1200
python -m conRWKV.benchmark.benchmark --model rwkv --backend RWKV-Runner --request-rate 4 --num-prompts 1200
```

### todolist

- 正确性验证
- 缩短ITL, TPOP

### acknowledgement

sonta老师、Zhiyuan Li老师的fla太强了

感谢PENG bo老师的rwkv v7, 期待更大的v7和v8