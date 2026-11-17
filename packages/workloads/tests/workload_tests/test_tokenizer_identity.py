"""GLM next-token geometry, rank ownership and exact tokenizer identity."""

import json


def test_prepared_corpus_records_checkpoint_tokenizer_identity(tmp_path):
    from shadowspill_workloads.inputs.text.corpus import write_meta
    from shadowspill_workloads.inputs.text.tokenizer import tokenizer_identity

    class Tokenizer:
        eos_token_id = 3

        def __len__(self):
            return 4

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text('{"vocab": ["a", "b", "c"]}')
    config = checkpoint / "tokenizer_config.json"
    config.write_text('{"eos_token": "c"}')
    counts = dict(files=["text.parquet"], train_tokens=256, val_tokens=128)
    meta = write_meta(tmp_path, Tokenizer(), str(checkpoint), counts)
    assert meta["tokenizer_sha256"] == tokenizer_identity(checkpoint)
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    # A changed special-token policy must fail the recipe's exact identity check.
    config.write_text('{"eos_token": "b"}')
    assert meta["tokenizer_sha256"] != tokenizer_identity(checkpoint)
