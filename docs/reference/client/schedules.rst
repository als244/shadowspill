Learning rate schedules
=======================

A schedule gives the learning rate for a zero-based step; pass its value with
each update, as in ``training.step(batch, optim={"lr": schedule(step)})``.

.. autoclass:: shadowspill_client.schedules.WarmupCosine

.. autoclass:: shadowspill_client.schedules.Constant
