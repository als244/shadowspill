# Live command table

`CommandTable` is an optional client display. Add each submitted future to the
table; it creates a pending row immediately and fills in results on completion.
It returns the same future and leaves submission order and batching to your loop.

```python
from shadowspill_client.display import CommandTable
from shadowspill_client.text import text_batch_statistics

table = CommandTable()
stats = text_batch_statistics(batch)
pending = training.step(batch, optim={"lr": 0.00001})
table.track("step", pending, batch=stats)

# Prepare and submit the next batch before waiting when pipelining.
result = pending.result()
```

The table shows command number, operation, submission and completion times, status,
loss, server execution time, estimated time when supplied, real tokens per second,
token/sequence counts and total/supervised sequence lengths. Lengths use
`min/max avg/med`, for example `345/3,507 1,126/1,013`.

Times in **Submitted** and **Completed** use your laptop's local clock; the timezone
appears above the table. **Time** comes from `result.execution_seconds`, excluding
client preparation, uploads, queueing and final artifact publication. **Tokens/s**
divides logical input tokens by that server duration. It excludes packing padding
and replicated worker copies.

## Split updates and evaluation

Track every command separately. `optim_step()` has no batch information or forward
loss. `forward()` shows loss only if its prepared output set returns it.

```python
backward = training.forward_backward(batch)
table.track("forward_backward", backward, batch=text_batch_statistics(batch))

update = training.optim_step(optim={"lr": 0.00001})
table.track("optim_step", update)

evaluation = training.forward(eval_batch)
table.track("forward", evaluation, batch=text_batch_statistics(eval_batch))
evaluation.result()
```

Prepare these operations during setup before using them; see the
[training workflow](training-workflow.md). The display does not prepare or submit
additional operations. Split losses remain the contributions returned by each
call, with your original weighting; the table does not renormalize them.

## Optional statistics and predictions

`text_batch_statistics` reads the aligned `tokens`, `target_tokens` and optional
`loss_mask` fields. It counts unmasked targets independently of objective weights.
Input-only evaluation has no supervised lengths. For other input formats, supply
`BatchStatistics.from_lengths(lengths, supervised=supervised_lengths)` or omit
batch statistics. No input tensors are retained by the display.

To show the prediction for the selected layout, validate the batch locally first:

```python
validation = training.validate_batch(batch, operation="forward_backward").result()
pending = training.forward_backward(batch)
table.track(
    "forward_backward", pending, batch=text_batch_statistics(batch),
    planned_seconds=validation.estimated_seconds,
)
```

`estimated_seconds` comes from the matching candidate's saved simulation. It is
absent when packing requires a server-side grouping callback. An optimizer-only
call should use its own saved Plan estimate, if supplied, rather than a whole-step
prediction. Omission displays a dash.

Terminals update recent rows in place; `max_rows=20` bounds retained display rows.
Redirected output appends submission and completion lines with the same command
number, without terminal control codes. A broken output stream disables rendering
and sets `table.error`; it does not cancel or retry training. Results and failures
still belong to the original futures.
