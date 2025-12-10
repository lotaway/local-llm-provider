import torch
from transformers import StoppingCriteria, TextIteratorStreamer
from typing import cast


class CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        t = torch.tensor(
            [self.cancelled] * batch_size, dtype=torch.bool, device=input_ids.device
        )
        return cast(torch.BoolTensor, t)

    def cancel(self):
        self.cancelled = True


class CancellableStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.stopping_criteria = CancellationStoppingCriteria()

    def cancel(self):
        self.stopping_criteria.cancel()
