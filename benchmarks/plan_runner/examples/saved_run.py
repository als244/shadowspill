"""Edit these paths, then run this short script on the allocated node."""

from benchmarks.plan_runner import benchmark

if __name__ == "__main__":
    result = benchmark(
        "runs/selected",  # Public training run with plans/training.ssp.
        output_dir="benchmarks/selected",  # A new directory for this attempt.
        steps=7,
        warmup_steps=1,  # Execute all seven; report the median of updates 2-6.
        trace_final_step=True,  # Default: save each rank's final update trace.
        # Optional: import NsysCapture from benchmarks.plan_runner, then add
        # nsys=NsysCapture(start_step=5, steps=2),  # Time 2-4; Nsight 5-6.
        # Set the GLOBAL work per update to also report throughput:
        # units_per_step=2_097_152, unit_label="tokens",
    )
    print(f"Measured: {result['median_seconds']:.3f} s/update")
    print(f"Predicted: {result['predicted_seconds']:.3f} s/update")
