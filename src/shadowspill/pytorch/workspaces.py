"""Initialize PyTorch's retained BLAS workspace on the current CUDA stream."""

import torch


def initialize_blas(device: torch.device) -> None:
    with torch.cuda.device(device), torch.no_grad():
        torch.cuda.current_blas_handle()  # type: ignore[no-untyped-call]
        left = torch.empty((2048, 2048), dtype=torch.bfloat16, device=device)
        right = torch.empty_like(left)
        output = torch.empty_like(left)
        try:
            torch.mm(left, right, out=output)
            torch.cuda.current_stream(device).synchronize()
        finally:
            del output, right, left
            torch.cuda.current_stream(device).synchronize()
