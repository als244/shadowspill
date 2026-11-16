# ShadowSpill client

Submit training batches to a ShadowSpill service from Python. The client needs
Python 3.12+ and NumPy; model and GPU libraries run on the server.

From a source checkout:

```bash
python -m pip install ./packages/client
```

`ServiceClient` connects and prepares a run. `TrainingClient` submits updates,
evaluation and checkpoint requests. Every operation returns a future; use
`.result()` to wait for its result.

Start with the [complete training example](../../docs/examples/client-training.md)
and [client guide](../../docs/python/client.md). See
[advanced usage](../../docs/python/advanced/service-client.md) for saved Plans,
reconnect, downloads, uploads and optional local planning.
