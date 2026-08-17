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
object through `shared_input()`.

One callable has one outstanding submitted invocation. Resolve its pending
result before reusing that callable. This keeps its admitted physical layout,
task records, and completion event single-owner while allowing separately
planned callables to be active together.
