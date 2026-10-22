"""Independent fixed-shape inputs for the training API."""


def split_microbatches(pair, *, size, sequence_length):
    """Keep each position's data while giving equivalent positions one contract.

    A slice into the full update retains its storage offset and backing storage.
    Both matter to capture guards. Even a contiguous slice can retain them, so
    clone the inputs just as the text quickstart does.
    """

    tokens, targets = pair
    lengths = (sequence_length,) * (size // sequence_length)
    for start in range(0, tokens.shape[1], size):
        yield (
            (
                tokens[:, start : start + size].clone(),
                targets[:, start : start + size].clone(),
                lengths,
            ),
            1.0,
        )
