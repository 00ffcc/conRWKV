# config.py
class Config:
    def __init__(self):
        self.port = 8000
        self.device = "cuda:0"
        self.max_seq_len = int(1e6)
        self.max_batch_size = int(1e6)
        self.max_queue_size = int(1e6)
        self.max_completion_tokens = 1024
        self.model = r"./weights/v7-1.5b.pth"

    def update(self, parsed_args):
        self.port = parsed_args.port
        self.device = parsed_args.device
        self.max_seq_len = parsed_args.max_seq_len
        self.max_batch_size = parsed_args.max_batch_size
        self.max_queue_size = parsed_args.max_queue_size
        self.max_completion_tokens = parsed_args.max_completion_tokens
        self.model = parsed_args.model

config = Config()