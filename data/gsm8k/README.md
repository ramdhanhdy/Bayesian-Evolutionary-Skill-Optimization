# GSM8K Local Data

Local JSONL files in this directory are generated from the Hugging Face
`openai/gsm8k` parquet files and are intentionally ignored by Git.

- `train.jsonl`: 90% of the official training split.
- `validation.jsonl`: deterministic 10% holdout from the official training
  split using pandas `sample(frac=0.10, random_state=42)`.
- `test.jsonl`: untouched official test split.

Configure the mini experiment with:

```env
BESO_GSM8K_TRAIN_JSONL=data/gsm8k/train.jsonl
BESO_GSM8K_VALIDATION_JSONL=data/gsm8k/validation.jsonl
BESO_GSM8K_TEST_JSONL=data/gsm8k/test.jsonl
```

Source:
https://huggingface.co/datasets/openai/gsm8k/tree/main/main
