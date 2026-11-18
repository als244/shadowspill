"""Which of a schema's capacities a step's token throughput counts."""

from shadowspill_client.protocol.inputs import InputSchema


def token_capacity(schema: InputSchema) -> str | None:
    """The one scheduled capacity the schema counts in tokens, if it has one.

    A schema names each capacity's unit. Throughput in tokens is defined only
    when a step schedules exactly one capacity in that unit; a schema with none,
    or with several, has no single token count to divide by a step's time.
    """
    names = [
        capacity.name
        for capacity in schema.capacities
        if capacity.scheduled and capacity.unit == "tokens"
    ]
    return names[0] if len(names) == 1 else None
