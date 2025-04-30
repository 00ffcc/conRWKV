import asyncio
import time
from typing import List, Optional, Union, Dict
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from conRWKV.utils import ChatCompletionRequest, generate_logits_processor, output_buffer, log

from transformers.generation.logits_process import LogitsProcessorList

from contextlib import asynccontextmanager
from conRWKV.config import config  # 导入 config 对象

@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task, request_queue
    request_queue = asyncio.Queue(maxsize=config.max_queue_size)
    load_model()
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


# Global variables
model = None
tokenizer = None
request_queue = None
background_task = None
device = None
# args = None
from conRWKV.models.v7.model_fla import RWKV
from conRWKV.tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
def load_model():
    global model, tokenizer, device
    device = config.device
    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tokenizer=RWKVWorldTokenizer(vocab_file=rf"{dir_path}/tokenizer/rwkv_vocab_v20230424.txt")
    model = RWKV.from_pretrained(config.model, device=device)
    model.eval()
    

# Background task for processing requests from the queue
async def process_request_queue():
    while True:
        try:
            batch = []
            # input_ids的总长度不能超过args.max_seq_len
            for _ in range(config.max_batch_size):
                try:
                    request_item = await asyncio.wait_for(request_queue.get(), timeout=0.01)  # Non-blocking get with timeout
                    sum_len = sum(len(request_item["input_ids"][0]) for request_item in batch)
                    if sum_len + len(request_item["input_ids"][0]) > config.max_seq_len:
                        request_item["next_input_ids"] = request_item["input_ids"][:, config.max_seq_len-sum_len:]
                        request_item["input_ids"] = request_item["input_ids"][:, :config.max_seq_len-sum_len]
                    batch.append(request_item)
                    if sum_len + len(request_item["input_ids"][0]) == config.max_seq_len:
                        break
                except asyncio.TimeoutError:
                    break  # No more requests in the queue

            if not batch:
                await asyncio.sleep(0.03)  # Avoid busy-waiting if the queue is empty
                continue
            log.info(f"processing batch of {len(batch)} requests...")
            
            # 1. Prepare the batch
            input_ids, states = prepare_batch(batch)
            log.info("batch prepared")
            
            # 2. Inference Step (Continuous Batching)
            try:
                with torch.no_grad():
                    logits, states = model(input_ids, state=states)     
            except Exception as e:
                # Handle model inference errors.  Important to catch and log.
                log.error(f"Inference error: {e}")
                for request_item in batch:
                    request_item["exception"] = e
                continue
            log.info("inference done")  


            # 3. Sampling Step (Token Generation) and Post-processing
            
            for i, request_item in enumerate(batch):
                request_item['all_ids'] = torch.cat([request_item['all_ids'], request_item['input_ids']], dim=1)
            new_input_ids, completions = sample_and_postprocess(logits, batch)
            log.info("sampling done")  
            
            
            # 4. Handle Streaming and Re-enqueue unfinished requests
            # states = states.to('cpu') TODO: state offload to cpu, offload的话很慢，不offload的话会爆显存
            
            for i, request_item in enumerate(batch):
                if request_item["next_input_ids"] is not None:
                    # chunk处理，仍然处于prefill阶段
                    request_item["input_ids"] = request_item["next_input_ids"]
                    request_item["next_input_ids"] = None
                    request_item["state"] = states[:, i:i+1]
                    request_queue.put_nowait(request_item)
                else:
                    # Stream and re-enqueue
                    if_stop = request_item['buffer'].update(completions[i], new_input_ids[i][0, 0].item())
                    if not if_stop:
                        # Update input_ids and re-enqueue
                        request_item["input_ids"] = new_input_ids[i]
                        request_item["state"] = states[:, i:i+1]
                        request_queue.put_nowait(request_item)
            log.info("requests processed")
        except Exception as e:
            log.error(f"Error: {e}")
            raise e
def prepare_batch(batch: List[Dict]):
    input_ids = [request_item["input_ids"] for request_item in batch]
    states = torch.cat([request_item.get("state", model.empty_state(1, device=device)).to(device) for request_item in batch], dim=1) # move states from cpu to gpu
    return input_ids, states


def sample_and_postprocess(logits: torch.Tensor, batch: List[Dict]):

    new_input_ids = [None] * len(batch)
    completions = [""] * len(batch)

    for i in range(len(batch)):
        request_item = batch[i]
        next_token = sample(logits[i], request_item['all_ids'], batch[i]['logits_processor'])
        # Decode the new token
        new_token = tokenizer.decode(next_token.squeeze(0)) # rwkv tokenizer only support dim=1
        # Update the completion
        completions[i] = new_token

        new_input_ids[i] = next_token

    return new_input_ids, completions


def sample(logits: torch.Tensor, input_ids: torch.Tensor, logits_processor: LogitsProcessorList) -> torch.Tensor:

    # Apply logits processors (temperature, top_p, frequency_penalty, etc.)
    
    
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
    log.critical(request.model_dump_json())
    if request.messages:
        # Prepare the prompt
        chat_mode = True
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=request.add_generation_prompt)
    else:
        chat_mode = False
        prompt = request.prompt
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)


    async def stream_generator():
        stream_queue:asyncio.Queue = asyncio.Queue()
        request_item = {
                            "request": request, 
                            "input_ids": input_ids, 
                            "next_input_ids": None, # 用于分chunk处理
                            "all_ids": torch.empty((1, 0), dtype=torch.int32, device=device), # 用于sample
                            "logits_processor": generate_logits_processor(request),
                            "buffer": output_buffer(
                                stop=request.stop, 
                                max_completion_tokens=request.max_completion_tokens, 
                                include_stop_str_in_output = request.include_stop_str_in_output, 
                                stream_queue=stream_queue),
                        }
        try:
            request_queue.put_nowait(request_item)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Server busy, try again later")
        
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
    # global args
    import uvicorn
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--max_seq_len", type=int, default=int(1e6), help="Max sequence length for input_ids")
    parser.add_argument("--max_batch_size", type=int, default=int(1e6), help="Max batch size for inference, to avoid OOM")
    parser.add_argument("--max_queue_size", type=int, default=int(1e6), help="Max queue size for requests, to avoid OOM")
    parser.add_argument("--max_completion_tokens", type=int, default=1024, help="Max number of tokens to generate")
    parser.add_argument("--model", type=str, default=r"./weights/v7-1.5b.pth", help="path to model weights")
    parsed_args = parser.parse_args()
    config.update(parsed_args)  # 更新 config
    torch.cuda.set_device(config.device)  # 使用 config
    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")
if __name__ == "__main__":
    main()