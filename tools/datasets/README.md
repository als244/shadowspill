# Prepare tokenized datasets

Run these commands from the source checkout with the dataset and tokenizer
libraries installed. They write client-owned token files and metadata; they do
not load model weights or start training.

```bash
python -m tools.datasets.prepare_chat --help
python -m tools.datasets.prepare_text --help
```

`prepare_chat` renders conversations with the tokenizer's chat template and writes
assistant-token masks for fine-tuning. `prepare_text` tokenizes ordinary text for
next-token pretraining. Both record tokenizer identity in `meta.json` alongside
`train.bin` and `val.bin`.

The [training demos](../../examples/demos/README.md) show complete dataset commands
and submit variable-length batches through the public client.
