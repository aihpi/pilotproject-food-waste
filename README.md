---
language: en
tags:
- vision-language
- food-waste
- qwen
- lora
license: apache-2.0
---

# Food Waste Vision-Language Model

This is a fine-tuned version of Qwen-VL model for food waste detection and analysis. The model has been trained using LoRA fine-tuning on a custom food waste dataset.

## Model Details

- **Base Model**: Qwen-VL
- **Fine-tuning Method**: LoRA
- **Task**: Vision-Language Understanding for Food Waste
- **License**: Apache 2.0

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "aihpi/food-waste-vlm"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Training Details

The model was fine-tuned using LoRA on a custom food waste dataset. The training was conducted using the LLaMA-Factory framework.

## License

This model is licensed under the Apache 2.0 License.