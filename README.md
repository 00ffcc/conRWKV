<h1 align="center">conRWKV</h1>

<div align="center">
    本项目旨在提供一个高并发的 RWKV 云端推理引擎，以方便后续基于 RWKV 的应用。
</div>

### News

- 4.24 fla版本切换至主线 (可能产生性能差异，测试中)
- ~3.2 部署了一个免费的 RWKV v7 2.9B API！地址 1.92.129.93(域名备案中)，欢迎来玩！~ 暂时没了
- 3.3 用 API 测了大海捞针，结果很有意思:)

### Feature

- 支持 continuous batching，确保高吞吐量。
- prefill 使用[fla(flash linear attention)](https://github.com/fla-org/flash-linear-attention/)，使用tensor core加速prefill，降低TTFT(Time to First Token)。
- 兼容 OpenAI API。
- 支持 RWKV v7。
- 极简安装部署

### Install & Deploy

```bash
pip install git+https://github.com/00ffcc/conRWKV.git@master
conRWKV --model /path/to/pth
```

### Benchmark

这里展示了与 [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) 的比较。

由于服务器上没有 vulkan 环境，暂时无法测试 [Ai00](https://github.com/Ai00-X/ai00_server)；由于 [RWKV-Infer](https://github.com/OpenMOSE/RWKV-Infer) 不支持文本补全，也暂时无法测试。有测试的方法请戳我。

| RPS  | Num prompts | Engine      | Successful requests | Median E2E Latency | Median TTFT | Median ITL |
| ---- | ----------- | ----------- | ------------------- | ------------------ | ----------- | ---------- |
| 2    | 1200        | conRWKV     | **1200**            | 7750.72            | **115.76**  | 56.33      |
| 2    | 1200        | RWKV-Runner | **189**[^1]         | 3746.53            | **2176.03** | 16.29      |
| 4    | 1200        | conRWKV     | **1200**            | 10469.67           | **145.69**  | 77.45      |
| 4    | 1200        | RWKV-Runner | **119**[^1]         | 4484.30            | **1633.34** | 15.78      |

[^1]: RWKV-Runner 是逐个处理请求，并发时失败概率较大。

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

支持 OpenAI api 协议中的常用参数。
- **model**: 填什么都行，并没有什么意义。
- **messages/prompt**: messages 用于 chat 模式，prompt 用于续写模式。为了方便起见使用同一个 url。
- **frequency_penalty**: 频率惩罚。
- **max_completion_tokens/max_tokens**: 最多生成的 token 数量，优先使用 max_completion_tokens。
- **n**: 目前只支持 n=1。
- **best_of**: 目前只支持 best_of=1。
- **seed**: 尚未实现。
- **stop**: 默认为 `['\n\nUser', '\n\nUser:', 'User:']`，因为 rwkv v7 容易忘记在 `User:` 前加入 `\n\n`。（SFT 的问题？）
- **stream**: 是否流式返回。
- **temperature**: 温度，用于采样。
- **top_p**: 用于采样。
- **ignore_eos**: 是否忽略 stop, 用于 benchmark 测试。
- **include_stop_str_in_output**: 是否在输出中包含 stop 里的内容。
- **add_generation_prompt**: 是否要加上 `\n\nAssistant:`

### ToDoList

- 正确性验证
- 缩短 ITL, TPOP
- 支持更多参数

### Acknowledgement

sonta 老师、Zhiyuan Li 老师的 fla 太强了

感谢 PENG bo老师的 RWKV-v7, 期待更大的 v7 和 v8 模型
