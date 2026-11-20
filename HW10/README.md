# HW10: TinyLlama 1.1B + LoRA for QA bug reports

This version uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, a causal chat model, with LoRA on `q_proj` and `v_proj`. It uses the model's chat template and trains four independent output-field tasks (Category, Title, Expected, Actual). The dataset consists of 60 unique bug scenarios; only training data are augmented.

## Run on Apple Silicon

```bash
conda activate hw10_qa
python -m pip install -r requirements.txt
python train.py
```

The model download is roughly several GB and requires significantly more RAM and time than DistilGPT2. This version uses float32 (no bitsandbytes/8-bit quantization), batch size 1, gradient accumulation 8, and gradient checkpointing. On an 8 GB Mac it may still run out of unified memory; 16 GB or more is preferable. If memory errors occur, stop training and send the error log. Do not assume it will complete on all Mac configurations.

Outputs are written to `outputs_tinyllama/`. Results are not guaranteed; compare metrics and manually review text. Accuracy measures exact category extraction, while field completeness only checks for nonempty strings.
