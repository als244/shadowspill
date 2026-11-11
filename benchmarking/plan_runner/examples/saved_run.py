"""Edit these paths, then run this short script on the allocated node."""

from benchmarking.plan_runner import benchmark

if __name__ == "__main__":
    result = benchmark(
        "runs/selected",  # Public training run with plans/training.ssp.
        output_dir="benchmarks/selected",  # A new directory for this attempt.
        steps=7,
        warmup_steps=1,  # Execute all seven; report the median of updates 2-7.
        # Set the GLOBAL work per update to also report throughput:
        # units_per_step=2_097_152, unit_label="tokens",
    )
    print(f"Measured: {result['median_seconds']:.3f} s/update")
    print(f"Predicted: {result['predicted_seconds']:.3f} s/update")
