# Finetuning Gemma 2B Model

This repository contains a single notebook that fine-tunes `google/gemma-2b` using LoRA adapters on a quote-generation dataset, with 4-bit NF4 quantization to reduce VRAM usage.

The core file is `finetuning_Gemma_2b_model.ipynb`.

## Project Summary

- Objective: adapt Gemma 2B to generate quote and author pairs in the style of the `Abirate/english_quotes` dataset.
- Base model: `google/gemma-2b`.
- Fine-tuning method: LoRA (parameter-efficient fine-tuning) with `trl.SFTTrainer`.
- Quantization: 4-bit NF4 (`bitsandbytes`) with compute dtype `torch.bfloat16`.
- Dataset: `Abirate/english_quotes` (train split only, 2508 rows).
- Notebook runtime metadata: Google Colab GPU runtime (`T4` listed in metadata).
- License: MIT (`LICENSE`).

## Repository Structure

- `finetuning_Gemma_2b_model.ipynb`: complete training/inference workflow.
- `LICENSE`: MIT license text.

## What The Notebook Does End-to-End

1. Installs required libraries.
2. Imports model/training utilities.
3. Reads Hugging Face token from Colab Secrets.
4. Configures 4-bit NF4 quantization.
5. Loads tokenizer and base Gemma model.
6. Runs baseline generations before fine-tuning.
7. Configures LoRA adapters.
8. Loads and inspects the quote dataset.
9. Defines formatting function for supervised fine-tuning text.
10. Builds `SFTTrainer` with training hyperparameters.
11. Trains for 100 steps.
12. Runs post-training generations to show behavior change.

## Notebook Cell-By-Cell Analysis

### Cell 1: Install Dependencies

```python
!pip install -q --upgrade pip
!pip install -q transformers accelerate peft trl bitsandbytes datasets
```

Purpose:
- Upgrades `pip`.
- Installs required training stack.

Packages used:
- `transformers`: model and tokenizer loading.
- `accelerate`: distributed/runtime integration.
- `peft`: LoRA adapter definitions.
- `trl`: `SFTTrainer` for supervised fine-tuning.
- `bitsandbytes`: 4-bit quantization and 8-bit optimizer.
- `datasets`: Hugging Face dataset loading.

### Cell 2: Imports

```python
import os
import torch
from datasets import load_dataset
import transformers
from google.colab import userdata
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GemmaTokenizer
from peft import LoraConfig
```

Notes:
- `userdata` ties this notebook to Colab Secrets.
- `GemmaTokenizer` is imported but not used directly; `AutoTokenizer` is used.

### Cell 3: Hugging Face Token Setup

```python
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```

Purpose:
- Pulls `HF_TOKEN` from Colab secret store and makes it available as an environment variable.

Why needed:
- `google/gemma-2b` is a gated model on Hugging Face and requires accepted terms and valid auth.

### Cell 4: Markdown Section Header

`# NF4( 4bit normalfloat)`

Purpose:
- Marks the quantization section.

### Cell 5: Quantization Configuration

```python
model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
     load_in_4bit=True,
     bnb_4bit_quant_type="nf4",
     bnb_4bit_compute_dtype=torch.bfloat16
 )
```

Purpose:
- Uses 4-bit quantized weights (NF4) for memory-efficient loading/training.

Important details:
- `load_in_4bit=True`: quantized weight storage.
- `bnb_4bit_quant_type="nf4"`: normal float 4-bit quantization.
- `bnb_4bit_compute_dtype=torch.bfloat16`: computation precision during operations.

### Cell 6: Model + Tokenizer Loading

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])
```

Purpose:
- Downloads tokenizer and model.
- Loads model directly onto GPU 0 with quantization enabled.

Notes:
- `device_map={"":0}` pins the model to a single GPU.
- Download progress outputs are present in notebook outputs.

### Cells 7-8: Baseline Generation Before Fine-Tuning

Prompts used:
- `Quote: I am not talented, but I'm good at not giving up`
- `Quote: Imagination is more`

Purpose:
- Shows base model behavior before domain adaptation.

Observed behavior:
- First output continues text but is verbose and repetitive.
- Second output includes HTML-style text (`<strong>`, `<h3>`) and repeated phrasing.

### Cell 9: W&B Environment Flag

```python
os.environ["WANDB_DISABLED"] = "false"
```

Purpose:
- Explicitly sets W&B integration as not disabled.

Note:
- This does not initialize W&B by itself; it only sets an env flag.

### Cell 10: LoRA Configuration

```python
lora_config = LoraConfig(
    r = 8,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM"
)
```

Purpose:
- Defines adapter rank and target projection layers for PEFT.

Meaning:
- `r=8`: adapter rank (capacity vs efficiency tradeoff).
- `target_modules`: attention and MLP projection layers in Gemma.
- `task_type="CAUSAL_LM"`: causal language modeling objective.

### Cell 11: Dataset Loading

```python
from datasets import load_dataset

data = load_dataset("Abirate/english_quotes")
```

Purpose:
- Pulls dataset from Hugging Face Datasets Hub.

Dataset details found in notebook:
- Features: `quote`, `author`, `tags`.
- Train rows: `2508`.

### Cell 12: Quick Data Inspection

```python
data['train']['quote']
```

Purpose:
- Displays examples from the quote column.

### Cell 13: Formatting Function For Trainer

```python
def formatting_func(example):
  quote_text = example['quote'][0] if isinstance(example['quote'], list) else example['quote']
  author_text = example['author'][0] if isinstance(example['author'], list) else example['author']
  text = f"Quote: {quote_text}\nAuthor: {author_text}"
  return text
```

Purpose:
- Converts dataset examples into supervised text format expected by `SFTTrainer`.

Output template:
- `Quote: <quote text>`
- `Author: <author name>`

Why list checks exist:
- Defensive handling for possible batched/list inputs vs scalar row inputs.

### Cell 14: Train Split Inspection

```python
data['train']
```

Purpose:
- Verifies dataset object and row count.

Observed output:
- `Dataset({ features: ['quote', 'author', 'tags'], num_rows: 2508 })`

### Cell 15: SFTTrainer Configuration

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
```

Purpose:
- Builds the training pipeline with quantized base model + LoRA adapters.

Key hyperparameters:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4` (effective batch accumulation)
- `warmup_steps=2`
- `max_steps=100`
- `learning_rate=2e-4`
- `bf16=True`, `fp16=False`
- `optim="paged_adamw_8bit"`
- `output_dir="outputs"`

Observed warnings in notebook output:
- PEFT warning indicates model was modified with PEFT more than once.
- This commonly happens when running adapter setup/trainer cells repeatedly in the same runtime without reloading model.

### Cell 16: Train

```python
trainer.train()
```

Observed training summary from notebook:
- `global_step=100`
- `train_loss=1.813469830751419`
- `train_runtime=268.8577`
- `train_samples_per_second=1.488`
- `train_steps_per_second=0.372`

### Cells 17-18: Post-Training Inference

Prompts used:
- `Quote: A Woman is like a tea bag;`
- `Quote: Outside of a dog, a book is man's`

Observed outputs:
- Output 1 completes quote and adds author:
  - `Quote: A Woman is like a tea bag; you never know how strong she is until you put her in hot water.`
  - `Author: Eleanor Roosevelt`
- Output 2 continues quote and starts author line:
  - `Quote: Outside of a dog, a book is man's best friend. Inside of a dog, it's too dark to read.`
  - `Author: Grou` (truncated due max token limit)

### Cell 19: Empty Cell

No code.

## Training Configuration At A Glance

- Model: `google/gemma-2b`
- Quantization: 4-bit NF4
- Compute dtype: `bfloat16`
- PEFT: LoRA rank 8
- Optimizer: `paged_adamw_8bit`
- Train steps: 100
- Data split: train only
- Logging interval: every step
- Output directory: `outputs`

## Reproducing The Notebook

## Prerequisites

- A GPU environment (Colab recommended for this exact notebook style).
- Hugging Face account with:
  - access approved for `google/gemma-2b`
  - token stored as Colab Secret key: `HF_TOKEN`

## Steps

1. Open `finetuning_Gemma_2b_model.ipynb` in Google Colab.
2. Enable GPU runtime.
3. Add `HF_TOKEN` in Colab Secrets.
4. Run cells in order from top to bottom.
5. Watch training logs and final inference outputs.

## Local Run Notes (if not using Colab)

- Replace `from google.colab import userdata` and secret access with your own token loading approach.
- Ensure CUDA, `bitsandbytes`, and compatible PyTorch versions are installed.

## Outputs And Artifacts

Produced during execution:
- Downloaded model/tokenizer cache in runtime cache dirs.
- Training artifacts under `outputs` directory in runtime.
- In-memory model updated with LoRA adapters.

Not currently included in notebook:
- Explicit `model.save_pretrained(...)` or adapter export cell.
- Explicit `tokenizer.save_pretrained(...)`.
- Push to Hub flow.

If runtime resets before saving, trained adapter state is lost.

## Known Limitations In Current Notebook

- No validation split or evaluation metrics (only training loss shown).
- No random seed set; exact reproducibility is limited.
- No checkpointing/save/export step in notebook cells.
- `WANDB_DISABLED` env var is set but no explicit W&B run initialization.
- PEFT warning can appear when re-running setup cells without resetting runtime.
- `bf16=True` can fail on some GPUs; if so, switch to `fp16=True` and `bf16=False`.

## Recommended Improvements

1. Add deterministic seeding (`transformers.set_seed(...)`).
2. Add eval split and metric tracking.
3. Save adapter and tokenizer after training.
4. Add optional `push_to_hub` workflow.
5. Add explicit `max_seq_length` and optionally packing in `SFTTrainer`.
6. Add inference helper function and standardized prompts for before/after comparison.
7. Add notebook markdown sections for setup, train, evaluate, save.

## Example Save Snippet To Add

```python
save_dir = "gemma-2b-lora-quotes"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

## Example Hub Push Snippet (Optional)

```python
# Requires huggingface_hub login and repo permissions
# trainer.model.push_to_hub("<username>/gemma-2b-lora-quotes")
# tokenizer.push_to_hub("<username>/gemma-2b-lora-quotes")
```

## Security And Access Notes

- Do not hardcode HF tokens in notebook cells.
- Keep using secret storage (`userdata.get('HF_TOKEN')`) or environment variables.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Status

Current state: prototype notebook demonstrating QLoRA-style fine-tuning flow for quote generation with Gemma 2B.
