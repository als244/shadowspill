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
