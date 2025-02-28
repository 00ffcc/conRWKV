from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Union, Dict
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)

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
        if self.max_completion_tokens is None:
            self.max_completion_tokens = self.max_tokens
        return self

    # 如果stop是str，则转为list。如果为None，则设置为[]
    @model_validator(mode='after')
    def check_stop(self):
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop is None:
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

def generate_stop_criteria_checker(request: ChatCompletionRequest):
    def stop_criteria_checker(generated_strs: str, token_count: int) -> bool:
        if any(stop_seq in generated_strs for stop_seq in request.stop):
            return True
        if token_count >= request.max_completion_tokens:
            return True
        return False
    return stop_criteria_checker
