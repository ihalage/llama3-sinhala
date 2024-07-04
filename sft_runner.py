"""
 ****************************************************************************

    @file       sft_runner.py

    @author     Achintha Ihalage

    @brief      This module is responsible for finetuning LLMs including LLaMA3
                and Phi3 for sinhala language understanding and responding in
                Sinhala. 

 ****************************************************************************
"""

import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from accelerate import PartialState
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

import wandb
wandb.init(project="LLaMA3_finetune_Phi3", name="phi3_sinhala")

class SinhalaSFT:
    """
	Args:
        model_id_or_path (str): Hugging Face model id or path to a saved model.
        dataset_id (str): Hugging Face dataset id.
        output_model_path (str): Path to save the trained LoRA adapters or the model if finetuned fully.
        learning_rate (float): Learning rate.
        batch_size (int): Per device batch size (for both training and evaluation)
        gradient_accumulation_steps (int): Gradient accumulation steps for simulating larger batch size.
        n_epochs (int): Number of epoch to finetune the model on.
        max_seq_len (int): Maximum sequence length for training. Must be less than the max_seq_len supported by the model.
        use_lora (bool): To use LoRA for parameter efficient finetuning.
        lora_r (int): Rank of the low rank decomposition.
        lora_alpha (int): Scaling factor for weight updates.
        lora_dropout (float): Dropout rate of LoRA adapters.
        lora_target_modules (list): List of target (e.g., linear) modules to apply LoRA adapters.
	"""
    def __init__(self,
                model_id_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                dataset_id: str = "ihalage/sinhala-instruction-finetune-large",
                output_model_path: str = "finetuned_models/llama3_8b",
                learning_rate: float = 1e-4,
                batch_size: int = 4,
                gradient_accumulation_steps: int = 4,
                n_epochs: int = 5,
                max_seq_len: int = 2048,
                test_size: float = 0.15,
                use_lora: bool = True,
                lora_r: int = 16,
                lora_alpha: int = 32,
                lora_dropout: float = 0.05,
                lora_target_modules: list = ['k_proj', 'q_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
                ) -> None:
        
        self.model_id_or_path = model_id_or_path
        self.dataset_id = dataset_id
        self.output_model_path = output_model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_epochs = n_epochs
        self.max_seq_len = max_seq_len
        self.test_size = test_size
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        ## training args
        self.save_strategy = "steps"
        self.save_steps = 1000
        self.save_total_limit = 4
        self.logging_steps = 1000
        self.eval_steps = 1000

        ## config for quantization
        self.quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
        
    def prepare_data_for_chat_format(self, dataset):
        """
        Function to prepare the dataset in the specific format required to finetune LLMs in a supervised finetuning manner (SFFT).
        This inlcudes wrapping the system, user and assistant messages in a specific format as follows.

        Args:
            dataset: Hugging Face dataset

        Returns:
            A dataset formatted for chat-based finetuning.
        """
        tokenizer_id = self.model_id_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if "phi3" in self.model_id_or_path.lower(): ## phi3 requires left padding
            tokenizer.padding_side = 'left'
        
        def create_message_column(row):
           
            messages = [
                        {"role": "user", "content": row["question_prompt"]},
                        {"role": "assistant", "content": row["response_prompt"]},
                        ]

            # return a dictionary with a 'messages' key and the 'messages' list as its value.
            return {"messages": messages}

        """
        This will create a dataset for CLM. user_prompt and the assistant_response will be concatenated with special tokens 
        such as <s><system>\nsystem_prompt\n<user>user_prompt\n<assistant>assistant_response\n<\s>.
        These special tokens are specific to the model.
        """
        def format_dataset_chatml(row):
            return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

        logger.info("Applying the chat format to the dataset ...")
        dataset_chatml = dataset.map(create_message_column)
        formatted_dataset_chatml = dataset_chatml.map(format_dataset_chatml)
        formatted_dataset_chatml = formatted_dataset_chatml.train_test_split(test_size = self.test_size, seed=42)
        print(formatted_dataset_chatml)
        return formatted_dataset_chatml
    
    def get_attn_implementation(self):
        """
        A function to check if bf16 precision is supported and choose flash_attention_2 implementation if possible for efficiency.

        Returns:
            Attention implementation and compute data type (e.g, bfloat16 or float16)
        """
        if torch.cuda.is_bf16_supported():
          compute_dtype = torch.bfloat16
          attn_implementation = 'flash_attention_2'
        else:
          compute_dtype = torch.float16
          attn_implementation = 'sdpa'
        return attn_implementation, compute_dtype
    
    def get_training_args(self):
        """
        Function to get training arguments wrapped as a TrainingArguments class.

        Returns:
            Training arguments.
        """
        args = TrainingArguments(
            output_dir=self.output_model_path,
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_eval_batch_size=self.batch_size,
            log_level="debug",
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=self.eval_steps,
            num_train_epochs=self.n_epochs,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            # report_to="wandb",
            # load_best_model_at_end=True,
            seed=42,
            )
        return args
    
    def finetune_model(self):
        """
        Function to finetune an LLM. This gets the model by model_id and performs data processing steps and initializes model training.
        Full finetuning of a model or soft finetuning with LoRA is supported.
        """
        dataset = load_dataset(self.dataset_id, split="train")
        dataset_chatml = self.prepare_data_for_chat_format(dataset)

        attn_implementation, compute_dtype = self.get_attn_implementation()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path, trust_remote_code=True, add_eos_token=True, use_fast=True)
        # padding token is set to the unknown token.
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.
        # ID of the padding token is set to the ID of the unknown token.
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        ## padding side is set to 'left' for phi3, meaning that padding tokens will be added to the left of the sequence.
        if "phi3" in self.model_id_or_path.lower():
            tokenizer.padding_side = 'left'
        
        model = AutoModelForCausalLM.from_pretrained(
                  self.model_id_or_path, torch_dtype=compute_dtype, trust_remote_code=True,
                  attn_implementation=attn_implementation,
                  quantization_config=self.quant_config,
                  device_map={"": PartialState().process_index},
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        args = self.get_training_args()

        if self.use_lora:
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha =self.lora_alpha,
                lora_dropout =self.lora_dropout,
                target_modules=self.lora_target_modules
            )
        else:
            peft_config = None

        # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device) ## this is not supported for 4bit and 8bit quantization

        trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset_chatml['train'],
                    eval_dataset=dataset_chatml['test'],
                    peft_config=peft_config,
                    dataset_text_field="text",
                    max_seq_length=self.max_seq_len,
                    tokenizer=tokenizer,
                    args=args,
                    # data_collator=data_collator,
                    # compute_metrics=self.compute_metrics,
            )

        trainer.train()
        trainer.save_model()
        wandb.finish()

if __name__=="__main__":
    ftobj = SinhalaSFT(
        model_id_or_path = "meta-llama/Meta-Llama-3-8B-Instruct",
        # model_id_or_path = "microsoft/Phi-3-mini-4k-instruct",
        dataset_id = "ihalage/sinhala-instruction-finetune-large",
        output_model_path = "finetuned_models/llama3_8b",
        learning_rate = 5e-5,
        batch_size = 1,
        gradient_accumulation_steps = 8,
        n_epochs = 5,
        max_seq_len = 2048,
        test_size = 0.1,
        use_lora = True,
        lora_r = 32,
        lora_alpha = 64,
    )

    ftobj.finetune_model()