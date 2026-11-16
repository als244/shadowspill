# Custom training definitions

Follow the [step-by-step guide](../../../docs/python/advanced/custom-definitions.md)
to install this server package and run its client. It supplies a small regression
model, a weighted objective and SGD through the public extension API.

`src/example_training` is the installed implementation. `server.py` enables it;
`train.py` uses only the lightweight client and can run on another machine.
