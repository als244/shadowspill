"""Initialize PyTorch's retained BLAS workspace on the current CUDA stream."""

import torch


def initialize_blas(device: torch.device) -> None:
    with torch.cuda.device(device), torch.no_grad():
        torch.cuda.current_blas_handle()  # type: ignore[no-untyped-call]
        # Exercise the library with minimal scratch space. The retained BLAS
        # workspace is charged separately; large temporary matrices would
        # impose an unrelated startup floor on otherwise admissible Plans.
        left = torch.empty((32, 32), dtype=torch.bfloat16, device=device)
        right = torch.empty_like(left)
        output = torch.empty_like(left)
        try:
            torch.mm(left, right, out=output)
            torch.cuda.current_stream(device).synchronize()
        finally:
            del output, right, left
            torch.cuda.current_stream(device).synchronize()
