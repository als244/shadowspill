"""Observe actual provider destination products without reading GPU contents."""

from collections import Counter


class DestinationObserver:
    def __init__(self):
        import triton
        from mlops.providers.sonicmoe import lora_destinations
        from quack import gemm_interface

        self.phase = "startup"
        self.products = []
        self.compiles = []
        original_product = lora_destinations._Destinations.product
        original_gemm = gemm_interface.gemm_add_out
        original_listener = triton.knobs.compilation.listener
        current = []

        def gemm(left, right, prior, *args, **kwargs):
            if current:
                row = current[-1]
                output = kwargs["out"]
                assert prior.data_ptr() == output.data_ptr() == row["pointer"]
                assert kwargs["beta"] == 1.0 and not kwargs["add_to_output"]
                row["gemm_calls"] += 1
            return original_gemm(left, right, prior, *args, **kwargs)

        def product(targets, index, *args, **kwargs):
            selected = targets.selected.get(index)
            if selected is None:
                return original_product(targets, index, *args, **kwargs)
            destination, scale = selected
            row = dict(
                phase=self.phase,
                output=index,
                shape=list(destination.shape),
                dtype=str(destination.dtype),
                pointer=destination.data_ptr(),
                scale=scale,
                canonical_b=targets.canonical_b,
                gemm_calls=0,
            )
            current.append(row)
            try:
                result = original_product(targets, index, *args, **kwargs)
                assert result.numel() == 0 and row["gemm_calls"] == 1, row
                self.products.append(row)
                return result
            finally:
                assert current.pop() is row

        def compilation(**event):
            self.compiles.append(
                dict(
                    phase=self.phase,
                    name=event["src"].name,
                    cache_hit=event["cache_hit"],
                )
            )
            if original_listener is not None:
                original_listener(**event)

        lora_destinations._Destinations.product = product
        gemm_interface.gemm_add_out = gemm
        triton.knobs.compilation.listener = compilation

    def report(self, steps=(0, 1, 2)):
        counts = Counter(row["phase"] for row in self.products)
        assert all(counts[f"invocation-{step}"] > 0 for step in steps), counts
        assert not self.compiles, self.compiles
        return dict(
            status="PASS_SELECTED_PRODUCTS_USE_EXISTING_C_AND_OUTPUT",
            counts=dict(counts),
            products=self.products,
            triton_compiles=self.compiles,
            limits=(
                "Checks actual selected GEMM destinations and empty consumed outputs. "
                "Triton events do not inventory all external CuTe compilation. "
                "Physical pointer reuse can change across planned spills or regions."
            ),
        )
