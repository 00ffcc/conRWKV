<h1 align="center">conRWKV</h1>

<div align="center">
    本项目旨在提供一个高并发的RWKV云端推理引擎，以方便后续基于RWKV的应用。
</div>

### news

- ~3.2 部署了一个免费的RWKV v7 2.9B API！地址1.92.129.93(域名备案中)，欢迎来玩！~ 暂时没了
- 3.3 用api测了大海捞针，结果很有意思:)

### feature

- 支持continuous batching，确保高吞吐量。
- prefill使用[fla(flash linear attention)](https://github.com/fla-org/flash-linear-attention/)，使用tensor core加速prefill，降低TTFT(Time to First Token)。
- 兼容OpenAI API。
- 支持RWKV v7。
- 极简安装部署

### install & deploy

```bash
pip install git+https://github.com/00ffcc/conRWKV.git@master
conRWKV --model /path/to/pth
```

**注意：** 为了防止triton在处理变长数据时反复重新编译kernel，conRWKV中使用的[fla](https://github.com/00ffcc/flash-linear-attention)与[官方版本](https://github.com/fla-org/flash-linear-attention/)略有不同，安装前需要删除原有的fla或新开一个虚拟环境。

### benchmark

这里展示了与[RWKV-Runner](https://github.com/josStorer/RWKV-Runner)的比较。由于服务器上没有vulkan环境，暂时无法测试[Ai00](https://github.com/Ai00-X/ai00_server)；由于[RWKV-Infer](https://github.com/OpenMOSE/RWKV-Infer)不支持文本补全，也暂时无法测试。有测试的方法请戳我。

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
### 支持的参数

支持OpenAI api协议中的常用参数。
- **model**: 填什么都行，并没有什么意义。
- **messages/prompt**: messages用于chat模式，prompt用于续写模式。为了方便起见使用同一个url。
- **frequency_penalty**: 频率惩罚。
- **max_completion_tokens/max_tokens**: 最多生成的token数量，优先使用max_completion_tokens。
- **n**: 目前只支持n=1。
- **best_of**: 目前只支持best_of=1。
- **seed**: 尚未实现。
- **stop**: 默认为 `['\n\nUser', '\n\nUser:', 'User:']`，因为rwkv v7容易忘记在 `User:` 前加入 `\n\n`。（SFT的问题？）
- **stream**: 是否流式返回。
- **temperature**: 温度，用于采样。
- **top_p**: 用于采样。
- **ignore_eos**: 是否忽略stop, 用于benchmark测试。
- **include_stop_str_in_output**: 是否在输出中包含stop里的内容。
- **add_generation_prompt**: 是否要加上 `\n\nAssistant:`

### todolist

- 正确性验证
- 缩短ITL, TPOP
- 支持更多参数

### acknowledgement

sonta老师、Zhiyuan Li老师的fla太强了

感谢PENG bo老师的rwkv v7, 期待更大的v7和v8
