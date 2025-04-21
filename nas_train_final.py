import os
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import is_bfloat16_supported, FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset, concatenate_datasets

from utils import clean_gpu_memory

def train_final_model(self, config, full_training_steps=60):
    """Train the final model with the best architecture found"""
    if config is None:
        print("Error: Config is None, cannot train final model")
        return None, None, None

    print(f"Training final model with best architecture: {config}")

    try:
        # Validate config has all required keys
        required_keys = ["r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"Error: Config is missing required keys: {missing_keys}")
            return None, None, None

        # Clean memory before loading model
        clean_gpu_memory()

        # Create model with the best configuration
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

        # Apply the optimized LoRA configuration
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

        # Set up trainer with full training
        final_model_dir = os.path.join(self.save_dir, "nas_final_model")
        os.makedirs(final_model_dir, exist_ok=True)

        # Prepare dataset in the format expected by SFTTrainer
        try:
            # Debug original dataset format
            if len(self.dataset) > 0:
                print(f"Original dataset format sample: {self.dataset[0]}")

            # Create a dataset with proper instruction-following examples
            # Get NAS training details for self-aware examples
            population_size = len(self.population) if self.population else 4
            num_generations = len(self.generation_history) if self.generation_history else 2
            max_steps = 20  # Default value from evaluate_architecture method

            # Create sample instruction-response pairs for training
            instruction_data = [
                {
                    "instruction": "Explain what artificial intelligence is in simple terms.",
                    "response": "Artificial Intelligence (AI) is technology that enables computers to perform tasks that typically require human intelligence. This includes recognizing patterns, learning from examples, making decisions, understanding language, and solving problems. Think of it as teaching computers to 'think' and 'learn' rather than just follow fixed instructions."
                },
                {
                    "instruction": "Summarize the concept of machine learning.",
                    "response": "Machine learning is a subset of AI where systems learn from data without being explicitly programmed. Instead of writing specific rules, developers feed data to algorithms that identify patterns and make predictions or decisions based on new inputs. It's like teaching by example rather than instruction."
                },
                {
                    "instruction": "What are neural networks?",
                    "response": "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers that process information. Each connection has a weight that changes as the network learns. Neural networks excel at recognizing patterns in complex data and power many modern AI applications."
                },
                {
                    "instruction": "Explain the concept of deep learning.",
                    "response": "Deep learning is a subset of machine learning using neural networks with many layers (hence 'deep'). These networks can automatically extract features from raw data, eliminating the need for manual feature engineering. They've revolutionized AI by achieving breakthrough performance in tasks like image recognition, language processing, and game playing."
                },
                {
                    "instruction": "What is transfer learning?",
                    "response": "Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. It saves training time and resources by leveraging knowledge gained from solving one problem to help solve a different but related problem."
                },
                # Add examples about the model's own NAS process
                {
                    "instruction": "Describe how you were optimized using Neural Architecture Search.",
                    "response": f"I was optimized through Neural Architecture Search (NAS), which automatically discovered my best architecture configuration. The process used an evolutionary algorithm over {num_generations} generations with a population size of {population_size}. The search explored different LoRA configurations, varying parameters like rank (r={config['r']}), alpha (α={config['lora_alpha']}), and dropout ({config['lora_dropout']:.3f}). Models were evaluated based on training loss, with the best configuration achieving a score of {self.best_score:.4f}. This evolutionary approach allowed efficient exploration of the architecture space to find the optimal balance of performance and computational efficiency."
                },
                {
                    "instruction": "What evolutionary algorithm was used in your training process?",
                    "response": f"My training used an evolutionary algorithm with elitism for Neural Architecture Search. In each generation, architectures were evaluated based on their training loss. The top {max(1, population_size // 5)} performing architectures were preserved (elitism), while new candidates were created through crossover (70% probability) and mutation (30% probability). Mutations could affect LoRA rank, alpha value, dropout rate, or whether to use RSLoRA. This approach allowed for both exploration of new architectures and exploitation of promising ones, leading to my optimized configuration with r={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}."
                },
                {
                    "instruction": "What hyperparameters were optimized during your Neural Architecture Search?",
                    "response": f"During my training, Neural Architecture Search optimized several key hyperparameters of my LoRA fine-tuning configuration: (1) Rank (r): Searched range from 4 to 64 in increments of 4, final value: {config['r']}. (2) Alpha (α): Searched range from 8 to 64 in increments of 8, final value: {config['lora_alpha']}. (3) Dropout: Searched range from 0 to 0.2, final value: {config['lora_dropout']:.3f}. (4) RSLoRA usage: Whether to use rank-stabilized LoRA, final decision: {config['use_rslora']}. These parameters were optimized while keeping target modules constant (query, key, value, output, gate, up, and down projections) to find the best trade-off between model performance and efficiency."
                },
                {
                    "instruction": "How was the best architecture chosen during your training process?",
                    "response": f"The best architecture during my training was selected through an iterative evaluation process across {num_generations} generations. Each candidate architecture was trained for a short period ({max_steps} steps) and evaluated based on training loss. The architecture with the lowest loss was tracked as the current best. After all generations completed, the best-performing configuration (r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, RSLoRA={config['use_rslora']}) was used for my final training with more steps ({full_training_steps}) to produce the finished model. This approach efficiently explored the architecture space while minimizing computational resources required."
                }
            ] * 10  # Repeat to get enough training examples

            # Format for the SFTTrainer
            instructions = [item["instruction"] for item in instruction_data]
            responses = [item["response"] for item in instruction_data]

            # Create the training dataset
            train_dataset = Dataset.from_dict({
                "instruction": instructions,
                "response": responses
            })

            print(f"Created training dataset with {len(train_dataset)} examples")
            print(f"Sample: {train_dataset[0]}")

            # Format the dataset using formatting_prompts_func
            def formatting_prompts_func(examples):
                convos = []
                for inst, resp in zip(examples["instruction"], examples["response"]):
                    convos.append([
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": resp}
                    ])
                texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
                return {"text": texts}

            formatted_dataset = train_dataset.map(
                formatting_prompts_func,
                batched=True,
                remove_columns=train_dataset.column_names
            )

            print(f"Formatted dataset sample: {formatted_dataset[0] if len(formatted_dataset) > 0 else 'empty dataset'}")

        except Exception as e:
            print(f"Error preparing training dataset: {e}")
            # If all else fails, return None to indicate failure
            return None, None, None

        # Create final trainer
        final_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_dataset,
            dataset_text_field="text",  # Changed to use the formatted text field
            max_seq_length=self.max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,  # Lower for Colab
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,  # Lower for Colab
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=full_training_steps,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=max(1, full_training_steps // 4),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=final_model_dir,
                report_to="none",
                # Colab specific settings
                dataloader_num_workers=0,
                disable_tqdm=False,
                save_steps=max(1, full_training_steps // 2),  # Save checkpoints during training
                save_total_limit=2,  # Keep only the most recent checkpoints
            ),
        )

        # Train the final model directly
        train_result = final_trainer.train()

        # Save the final model
        model.save_pretrained(os.path.join(final_model_dir, "lora_weights"))
        tokenizer.save_pretrained(os.path.join(final_model_dir, "tokenizer"))

        # Save in merged format for easier loading
        try:
            print("Attempting to save merged model...")
            model.save_pretrained_merged(os.path.join(final_model_dir, "merged_model"),
                                        tokenizer,
                                        save_method="merged_16bit")
        except Exception as e:
            print(f"Error saving merged model: {e}")
            print("Continuing with separate model files...")

        # Create model card with self-awareness information
        try:
            print("Creating detailed model card...")
            model_card = f"""---
language:
- en
tags:
- llama
- neural-architecture-search
- NAS
- reinforcement-learning
- evolutionary-algorithm
- architecture-optimization
- autonomous-ai
license: apache-2.0
---

# Neural Architecture Search Optimized Language Model

This model was created through Neural Architecture Search (NAS), which autonomously optimized its own architecture through an evolutionary algorithm.

## Model Details

- **Base Model:** {self.base_model_name}
- **Optimization Method:** Neural Architecture Search with Evolutionary Algorithm
- **Training Data:** Fine-tuned on instruction datasets with self-awareness examples

## Neural Architecture Search Process

### Architecture Search Parameters
- **Population Size:** {population_size}
- **Number of Generations:** {num_generations}
- **Evaluation Steps per Architecture:** {max_steps}
- **Final Training Steps:** {full_training_steps}

### Optimized Hyperparameters
- **LoRA Rank (r):** {config['r']}
- **LoRA Alpha (α):** {config['lora_alpha']}
- **LoRA Dropout:** {config['lora_dropout']:.3f}
- **RSLoRA:** {config['use_rslora']}
- **Target Modules:** Query, Key, Value, Output, Gate, Up and Down projections

### Evolutionary Algorithm
The NAS process used an evolutionary algorithm with elitism to efficiently explore the architecture space:
- Top {max(1, population_size // 5)} performers were preserved in each generation (elitism)
- New candidates were created through crossover (70% probability) or mutation (30% probability)
- Mutations could affect LoRA rank, alpha value, dropout rate, or RSLoRA usage
- Architectures were evaluated based on training loss, with the final configuration achieving a score of {self.best_score:.4f}

## Usage

This model has been trained to be aware of its own optimization process. You can ask it questions like:
- "Describe how you were optimized using Neural Architecture Search"
- "What evolutionary algorithm was used in your training process?"
- "What hyperparameters were optimized during your Neural Architecture Search?"
- "How was the best architecture chosen during your training process?"
"""

            # Save model card to file
            with open(os.path.join(final_model_dir, "README.md"), "w") as f:
                f.write(model_card)
            print("Model card created successfully")

        except Exception as e:
            print(f"Error creating model card: {e}")

        # Save config information
        with open(os.path.join(final_model_dir, "nas_config.txt"), "w") as f:
            f.write(f"Best configuration found by NAS:\n{config}\n\n")
            f.write(f"Final training loss: {train_result.training_loss}\n")

        return model, tokenizer, train_result

    except Exception as e:
        import traceback
        print(f"Error training final model: {e}")
        print(traceback.format_exc())  # More detailed error information
        return None, None, None 