"""How many ranks a run has, and one rank's share of a step."""


def ranks(options):
    """How many workers the run uses: the nodes' ranks added up, else
    ``ranks``, else one."""
    nodes = options.get("nodes")
    if nodes:
        count = sum(int(node.get("ranks", 1)) for node in nodes)
        if options.get("ranks", count) != count:
            raise ValueError("ranks must equal the nodes' ranks added up")
        return count
    count = options.get("ranks", 1)
    if type(count) is not int or count < 1:
        raise ValueError("ranks must be a positive integer")
    return count


def rank_options(options):
    """The options one rank plans and trains with: ``rank_tokens_per_step``
    when a recipe gives ranks unequal shares, else an equal share of the
    step's ``max_tokens_per_step``, whole sequences either way."""
    tokens = options.get("rank_tokens_per_step")
    if tokens is None:
        count = ranks(options)
        if options["max_tokens_per_step"] % (count * options["max_seq_len"]):
            raise ValueError(
                "max_tokens_per_step must divide into whole sequences for every rank"
            )
        tokens = options["max_tokens_per_step"] // count
    elif tokens % options["max_seq_len"]:
        raise ValueError("rank_tokens_per_step must be whole sequences")
    return {**options, "max_tokens_per_step": tokens}
