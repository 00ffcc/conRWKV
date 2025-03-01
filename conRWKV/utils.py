from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Union, Dict
from sympy import im
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)
import asyncio

# Request Data Model (OpenAI API format)
class ChatCompletionRequest(BaseModel):
    model: str
    messages: Optional[List[Dict[str, str]]] = None  # [{"role": "user", "content": "..."}]
    prompt: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None # if max_completion_tokens is not set, max_tokens is used for completion
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    seed: Optional[int] = None  # Could be useful for reproducibility. Not implemented here.
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    include_stop_str_in_output: Optional[bool] = False
    add_generation_prompt: Optional[bool] = True

    # add validation so user can't set temperature and top_p to 0 at the same time
    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, value, values):
        if value == 0.0 and values.data.get("top_p") == 0.0:
            raise ValueError("temperature and top_p cannot both be 0")
        return value

    @field_validator("top_p")
    @classmethod
    def check_top_p(cls, value, values):
        if value == 0.0 and values.data.get("temperature") == 0.0:
            raise ValueError("temperature and top_p cannot both be 0")
        return value
    
    # add validation so user can't set both messages and prompt or neither
    @model_validator(mode='after')
    def check_messages_prompt(self):
        if self.messages is not None and self.prompt is not None:
            raise ValueError("Both messages and prompt cannot be set at the same time")
        if self.messages is None and self.prompt is None:
            raise ValueError("Either messages or prompt must be set")
        return self
    # n>1 or best_of>1 is not supported yet
    @model_validator(mode='after')
    def check_n_best_of(self):
        if self.n > 1 or self.best_of > 1:
            raise ValueError("n>1 or best_of>1 is not supported yet")
        return self
    
    # 如果max_completion_tokens没有设置，则使用max_tokens作为最大的生成长度
    @model_validator(mode='after')
    def check_max_completion_tokens(self):
        if self.max_completion_tokens is None and self.max_tokens is not None:
            self.max_completion_tokens = self.max_tokens
        if self.max_completion_tokens is None:
            from conRWKV.main import args
            self.max_completion_tokens = args.max_completion_tokens
        return self

    # 如果stop是str，则转为list。如果为None，则设置为[] 或 ['\n\nUser'](根据是否是chat模式)
    @model_validator(mode='after')
    def check_stop(self):
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop is None:
            self.stop = [] if self.prompt is not None else ['\n\nUser']
        if self.ignore_eos:
            self.stop = []
        return self

def generate_logits_processor(request: ChatCompletionRequest):

    logits_processors = [
        TemperatureLogitsWarper(request.temperature) if request.temperature > 0 else None,
        TopPLogitsWarper(request.top_p) if request.top_p > 0 else None,
        RepetitionPenaltyLogitsProcessor(request.frequency_penalty) if request.frequency_penalty > 0 else None,
    ]

    # Filter out None values
    logits_processors = [processor for processor in logits_processors if processor is not None]
    logits_processor = LogitsProcessorList(logits_processors)

    return logits_processor


class output_buffer:
    # 缓冲输出，防止输出stop中的内容
    def __init__(self, stop: List[str], max_completion_tokens: int, include_stop_str_in_output: bool, stream_queue:asyncio.Queue):
        self.stop = stop
        self.max_completion_tokens = max_completion_tokens
        self.max_buffer_size = max(len(i) for i in stop)-1 if stop else 0
        self.include_stop_str_in_output = include_stop_str_in_output
        self.buffer: List[str] = []
        self.generated_ids = []
        self.stream_queue = stream_queue
    
    def update(self, token_str: str, token_id: int):
        self.buffer.append(token_str)
        self.generated_ids.append(token_id)

        # check stop criteria

        if any(stop_seq in ''.join(self.buffer) for stop_seq in self.stop):
            # 需要停止
            stop_string = None
            for stop_seq in self.stop:
                if stop_seq in ''.join(self.buffer):
                    stop_string = stop_seq
                    break
            if not self.include_stop_str_in_output:
                # 去掉stop_string
                buf = ''.join(self.buffer)
                self.buffer = [buf[:len(buf) - len(stop_string)]]
            
            # 输出缓冲区内容
            self.stream_queue.put_nowait(''.join(self.buffer))
            self.stream_queue.put_nowait(None) # 停止信号

            return True # 停止
        
        if len(self.generated_ids) >= self.max_completion_tokens:
            # 总长度够了
            self.stream_queue.put_nowait(''.join(self.buffer))
            self.stream_queue.put_nowait(None) # 停止信号

            return True # 停止

        while len(''.join(self.buffer)) > self.max_buffer_size:
            # 缓冲区满了
            self.stream_queue.put_nowait(self.buffer.pop(0)) # 弹出最早的元素

        return False # 继续