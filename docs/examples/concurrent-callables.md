# Concurrent planned callables

Distinct planned callables can be dispatched before either result is
synchronized:

```python
first_pending = first_forward.submit([first_input])
second_pending = second_forward.submit([second_input])

first_output = first_pending.result()
second_output = second_pending.result()

first_forward.close()
second_forward.close()
```

Each `submit()` performs the complete host dispatch and returns after recording
the callable's public completion event. `result()` waits for that event once.
The callables may share one runtime and may consume the same runtime-owned
object through `shared_input()`. They may also share one model: a training
step and a forward pass planned over the same model imported with
`import_model_state()` both run on its state, and the forward sees each update
the step has made -- which is how a training loop evaluates as it goes.

One callable has one outstanding submitted invocation. Resolve its pending
result before reusing that callable. This keeps its admitted physical layout,
task records, and completion event single-owner while allowing separately
planned callables to be active together.
