import asyncio
import time
from typing import List, Optional, Union, Dict
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator, model_validator

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task
    load_model()
    # 先跑一次，编译下kernel...
    # TODO: 还有能不能优雅点编译kernel
    with torch.no_grad():
        state = model.empty_state(1)
        for T in [1, 16, 32, 64]:
            input_ids = torch.zeros((1, T), dtype=torch.int32, device=device)
            model.forward([input_ids], state)
    background_task = asyncio.create_task(process_request_queue())
    yield
    if background_task:
        background_task.cancel()
        await background_task

app = FastAPI(title="LLM Backend with Continuous Batching",
                description="A backend for serving LLMs with continuous batching using Hugging Face models and OpenAI-compatible API.",
                version="0.1.0",
                middleware=[
                    Middleware(
                        CORSMiddleware,
                        allow_origins=["*"],  # Adjust as needed
                        allow_credentials=True,
                        allow_methods=["*"],
                        allow_headers=["*"],
                    ),
                ],
                lifespan=lifespan,
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


class SamplingParams(BaseModel):
    """Sampling parameters for text generation."""

    n: int = 1
    temperature: float = 1.0
    top_p: float = 0.0
    frequency_penalty: float = 0.0
    max_completion_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

# Global variables
model = None
tokenizer = None
request_queue = asyncio.Queue()
background_task = None
device = None
args = None
from conRWKV.models.v7.model_fla import RWKV
from conRWKV.tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
def load_model():
    global model, tokenizer, device
    device = args.device
    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tokenizer=RWKVWorldTokenizer(vocab_file=rf"{dir_path}/tokenizer/rwkv_vocab_v20230424.txt")
    model = RWKV.from_pretrained(args.model, device=device)
    model.eval()
    

# Background task for processing requests from the queue
async def process_request_queue():
    while True:
        try:
            # print(f"{time.time():.2f} waiting for requests...")
            batch = []
            # input_ids的总长度不能超过args.max_seq_len
            for _ in range(args.max_batch_size):
                try:
                    request_item = await asyncio.wait_for(request_queue.get(), timeout=0.01)  # Non-blocking get with timeout
                    sum_len = sum(len(request_item["input_ids"][0]) for request_item in batch)
                    if sum_len + len(request_item["input_ids"][0]) > args.max_seq_len:
                        request_item["next_input_ids"] = request_item["input_ids"][:, args.max_seq_len-sum_len:]
                        request_item["input_ids"] = request_item["input_ids"][:, :args.max_seq_len-sum_len]
                    batch.append(request_item)
                    if sum_len + len(request_item["input_ids"][0]) == args.max_seq_len:
                        break
                except asyncio.TimeoutError:
                    break  # No more requests in the queue

            if not batch:
                await asyncio.sleep(0.03)  # Avoid busy-waiting if the queue is empty
                continue
            print(f"{time.time():.2f} processing batch of {len(batch)} requests...")
            
            # 1. Prepare the batch
            input_ids, sampling_params_list, states = prepare_batch(batch)
            print(f"{time.time():.2f} batch prepared")
            
            # 2. Inference Step (Continuous Batching)
            try:
                with torch.no_grad():
                    logits, states = model(input_ids, state=states)     
            except Exception as e:
                # Handle model inference errors.  Important to catch and log.
                print(f"Inference error: {e}")
                for request_item in batch:
                    request_item["exception"] = e
                continue
            print(f"{time.time():.2f} inference done")  


            # 3. Sampling Step (Token Generation) and Post-processing
            
            for i, request_item in enumerate(batch):
                request_item['all_ids'] = torch.cat([request_item['all_ids'], request_item['input_ids']], dim=1)
            new_input_ids, completions, finished_requests = sample_and_postprocess(logits, batch, sampling_params_list)
            print(f"{time.time():.2f} sampling done")  
            
            
            # 4. Handle Streaming and Re-enqueue unfinished requests
            # states = states.to('cpu') TODO: state offload to cpu, offload的话很慢，不offload的话会爆显存
            print(f"{time.time():.2f} moving states to cpu done")
            for i, request_item in enumerate(batch):
                if i in finished_requests:
                    # Complete request
                    # put final result and None(end marker)
                    request_item["stream_queue"].put_nowait(completions[i])
                    request_item["stream_queue"].put_nowait(None)
                elif request_item["next_input_ids"] is not None:
                    # chunk处理，仍然处于prefill阶段
                    request_item["input_ids"] = request_item["next_input_ids"]
                    request_item["next_input_ids"] = None
                    request_item["state"] = states[:, i:i+1]
                    request_queue.put_nowait(request_item)
                else:
                    # Stream and re-enqueue
                    request_item["stream_queue"].put_nowait(completions[i])
                    # Update input_ids and re-enqueue
                    request_item["input_ids"] = new_input_ids[i]
                    request_item["state"] = states[:, i:i+1]
                    request_item["token_count"] = request_item["token_count"] + 1
                    request_queue.put_nowait(request_item)
            print(f"{time.time():.2f} requests processed")
        except Exception as e:
            print(f"Error: {e}")
            raise e
def prepare_batch(batch: List[Dict]):
    """
    Prepares a batch of requests for inference.

    Args:
        batch: A list of dictionaries, where each dictionary represents a request and contains:
            - "request": The ChatCompletionRequest object.
            - "input_ids": The input IDs for the request (torch.Tensor).

    Returns:
        A tuple containing:
            - A list of input IDs (torch.Tensor) for the entire batch.
            - A list of SamplingParams objects for each request in the batch.
            - A list of batched states
    """
    input_ids = [request_item["input_ids"] for request_item in batch]
    sampling_params_list = [create_sampling_params(request_item["request"]) for request_item in batch]
    states = torch.cat([request_item.get("state", model.empty_state(1, device=device)).to(device) for request_item in batch], dim=1) # move states from cpu to gpu
    return input_ids, sampling_params_list, states

def create_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
    """
    Creates a SamplingParams object from a ChatCompletionRequest.

    Args:
        request: The ChatCompletionRequest object.

    Returns:
        A SamplingParams object containing the sampling parameters.
    """

    return SamplingParams(
        n=request.n,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        max_completion_tokens=request.max_completion_tokens,
        stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None,
    )


def sample_and_postprocess(logits: torch.Tensor, batch: List[Dict], sampling_params_list: List[SamplingParams]):
    """
    Samples tokens from the logits and post-processes the generated tokens.

    Args:
        logits: The logits tensor from the model output.
        batch: A list of request contexts.
        sampling_params_list: A list of SamplingParams objects for each request.

    Returns:
        A tuple containing:
            - A list of new input IDs (torch.Tensor) for the requests that are not finished.
            - A list of completions (strings) for each request in the batch.  Empty strings for requests that are not yet finished.
            - A set of indices of requests that have finished generating.
    """

    new_input_ids = [None] * len(batch)
    completions = [""] * len(batch)
    finished_requests = set()

    for i in range(len(batch)):
        request_item = batch[i]
        sampling_params = sampling_params_list[i]
        next_token = sample(logits[i], request_item['all_ids'], sampling_params)
        # Decode the new token
        new_token = tokenizer.decode(next_token.squeeze(0)) # rwkv tokenizer only support dim=1
        # Update the completion
        completions[i] = new_token

        # Check for stop conditions
        stop_criteria_met = False
        if sampling_params.stop_sequences:
            for stop_seq in sampling_params.stop_sequences:
                if stop_seq and stop_seq in new_token:
                    stop_criteria_met = True
                    break
        if sampling_params.max_completion_tokens and request_item["token_count"] + 1 >= sampling_params.max_completion_tokens:
            stop_criteria_met = True

        # If the request is finished, add it to the finished_requests set
        if stop_criteria_met:
            finished_requests.add(i)
        else:
            # Otherwise, update the input IDs
            new_input_ids[i] = next_token

    return new_input_ids, completions, finished_requests


def sample(logits: torch.Tensor, input_ids: torch.Tensor, sampling_params: SamplingParams) -> torch.Tensor:
    """
    Samples a token from the logits using the specified sampling parameters.

    Args:
        logits: The logits tensor for the current request.
        input_ids: The input IDs for the current request.
        sampling_params: The SamplingParams object for the current request.

    Returns:
        A torch.Tensor containing the sampled token ID.
    """

    # Apply logits processors (temperature, top_p, frequency_penalty, etc.)
    
    logits_processors = [
        TemperatureLogitsWarper(sampling_params.temperature) if sampling_params.temperature > 0 else None,
        TopPLogitsWarper(sampling_params.top_p) if sampling_params.top_p > 0 else None,
        RepetitionPenaltyLogitsProcessor(sampling_params.frequency_penalty) if sampling_params.frequency_penalty > 0 else None,
    ]

    # Filter out None values
    logits_processors = [processor for processor in logits_processors if processor is not None]
    logits_processor = LogitsProcessorList(logits_processors)
    
    logits.unsqueeze_(0)  # Add a batch dimension

    processed_logits = logits_processor(input_ids, logits)

    logits.squeeze_(0)  # Remove the batch dimension

    # Sample the next token
    probs = torch.softmax(processed_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# API Endpoint
@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def chat_completions(request: ChatCompletionRequest, fastapi_request: Request):
    print(request.json())
    if request.messages:
        # Prepare the prompt
        chat_mode = True
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
    else:
        chat_mode = False
        prompt = request.prompt
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if request.max_completion_tokens is None:
        request.max_completion_tokens = request.max_tokens

    async def stream_generator():
        stream_queue:asyncio.Queue = asyncio.Queue()
        request_item = {
                            "request": request, 
                            "input_ids": input_ids, 
                            "next_input_ids": None, # 用于分chunk处理
                            "stream_queue": stream_queue, 
                            "token_count": 0,
                            "all_ids": torch.empty((1, 0), dtype=torch.int32, device=device), # 用于sample
                        }
        request_queue.put_nowait(request_item)

        if request.stream:
            while True:
                try:
                    completion = await stream_queue.get()
                    if completion is None:
                        yield "data: [DONE]".encode('utf-8')
                        break
                    data = json.dumps({
                        'id': 'cmpl-' + str(time.time()),
                        'object': 'chat.completion.chunk' if chat_mode else 'text_completion',
                        'created': int(time.time()),
                        'model': request.model,
                        'choices': [(
                            {
                                'delta': {'content': completion},
                                'index': 0,
                                'finish_reason': None,
                            } if chat_mode else
                            {
                                'text': completion,
                                'index': 0,
                                'finish_reason': None,
                            }
                        )],
                    })
                    yield (f"data: {data}\n\n").encode('utf-8')
                except asyncio.CancelledError:
                    break
        else:
            final_completion = ''
            while True:
                try:
                    completion = await stream_queue.get()
                    if completion is None:
                        break
                    final_completion += completion
                except asyncio.CancelledError:
                    break
            yield json.dumps({
                'id': 'cmpl-' + str(time.time()),
                'object': 'chat.completion.chunk' if chat_mode else 'text_completion',
                'created': int(time.time()),
                'model': request.model,
                'choices': [(
                            {
                                'message': {'role': 'assistant', 'content': final_completion},
                                'index': 0,
                                'finish_reason': None,
                            } if chat_mode else
                            {
                                'text': final_completion,
                                'index': 0,
                                'finish_reason': None,
                            }
                        )],
                }).encode('utf-8')

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

def main():
    global args
    import uvicorn
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--max_seq_len", type=int, default=1048576, help="Max sequence length for input_ids")
    parser.add_argument("--max_batch_size", type=int, default=128, help="Max batch size for inference, to avoid OOM")
    parser.add_argument("--model", type=str, default=r"./weights/v7-1.5b.pth", help="path to model weights")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
if __name__ == "__main__":
    main()