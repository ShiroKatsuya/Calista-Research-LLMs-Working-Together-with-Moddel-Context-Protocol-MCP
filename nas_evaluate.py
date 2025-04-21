import os
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import is_bfloat16_supported, FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, get_chat_template

from utils import clean_gpu_memory

# Evaluation function for NeuralArchitectureSearch class
def evaluate_architecture(self, config, max_steps=20):
    """Train and evaluate a single architecture configuration"""
    print(f"Evaluating architecture: {config}")
    
    # Create model with this configuration
    try:
        # Clean memory before loading new model
        clean_gpu_memory()
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
        
        # Apply the specific LoRA configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["r"],
            target_modules=config["target_modules"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=config["use_rslora"],
        )
        
        # Create a unique output dir for this evaluation
        config_id = f"r{config['r']}_a{config['lora_alpha']}_d{config['lora_dropout']:.2f}"
        output_dir = os.path.join(self.save_dir, f"eval_{config_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up trainer with smaller number of steps for quick evaluation
        eval_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="conversations",
            max_seq_length=self.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,  # Lower for Colab
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,  # Lower for Colab
                gradient_accumulation_steps=4,
                warmup_steps=1,
                max_steps=max_steps,  # Short evaluation
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=max(1, max_steps // 4),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
                report_to="none",
                # Colab specific settings for stability
                dataloader_num_workers=0,
                disable_tqdm=False,
            ),
        )
        
        # Apply chat template for proper training
        eval_trainer = train_on_responses_only(
            eval_trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        
        # Train and evaluate with error handling
        try:
            train_result = eval_trainer.train()
            eval_loss = train_result.training_loss
        except Exception as e:
            print(f"Training failed with error: {e}")
            eval_loss = float('inf')  # Penalize failed configurations
        
        # Free up memory
        del model, eval_trainer
        clean_gpu_memory()
        
        return eval_loss
        
    except Exception as e:
        print(f"Error during architecture evaluation: {e}")
        return float('inf')  # Return worst possible score for failed configurations 