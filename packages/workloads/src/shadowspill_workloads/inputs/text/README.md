# Text input adapters

`preparation.py` declares the token schema, packing capacities, runtime tensor
views and representative inputs shared by the supplied text architectures.
Sequence metadata remains input data for captured computation.

Corpus preparation lives in [dataset tools](../../../../../../tools/datasets/README.md).
Clients own dataset iteration and submit global `Datum` batches; the
[client demos](../../../../../../examples/demos/README.md) show this workflow.
Independent reference iterators live under
[qualification](../../../../../../qualification/reference/training/data/).
