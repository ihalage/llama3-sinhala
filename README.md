[![llama3-sinhala](https://img.shields.io/badge/llama3--sinhala-huggingface-orange)](https://huggingface.co/ihalage/llama3-sinhala)

# llama3-sinhala

LLaMA3 (8B) model instruction finetuned to understand and respond in Sinhala language. `meta-llama/Meta-Llama-3-8B-Instruct` is finetuned on a reletively large dataset in
Sinhala compiled by translating English datasets such as ELI5 and Alpaca. The dataset in hosted in Hugging Face Datasets hub [(`sinhala-instruction-finetune-large`)](https://huggingface.co/datasets/ihalage/sinhala-instruction-finetune-large)

The original model is 4-bit quantized and finetuned with a causal language modelling (CLM) objective by adding LoRA adapters with a rank of 16 and a scaling factor of 32.

The finetuned `llama3-sinhala` model generates better responses in Sinhala compared to the original instruction finetuned model released by Meta. The model is available in Hugging Face [(`llama3-sinhala`)](https://huggingface.co/ihalage/llama3-sinhala).
