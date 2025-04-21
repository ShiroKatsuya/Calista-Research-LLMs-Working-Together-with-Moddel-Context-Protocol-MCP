from unsloth import FastLanguageModel
import torch
import os
from google.colab import drive
import gc
from tqdm.auto import tqdm
import pickle
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

# Mount Google Drive for persistent storage
try:
    drive.mount('/content/drive')
    SAVE_DIR = '/content/drive/MyDrive/nas_llm_optimizer'
    os.makedirs(SAVE_DIR, exist_ok=True)
except:
    SAVE_DIR = './nas_outputs'
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Warning: Google Drive not mounted. Using local storage instead.")

# Function to clean up GPU memory
def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Select appropriate model size based on available GPU
def get_appropriate_model_name():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Available GPU memory: {gpu_memory:.2f} GB")
        
        if gpu_memory > 30:  # A100 40GB or better
            return "unsloth/Llama-3.2-3B-Instruct"
        elif gpu_memory > 12:  # T4 16GB
            return "unsloth/Llama-3.2-1B-Instruct"
        else:  # Smaller GPUs
            return "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    else:
        print("No GPU detected. Using smallest model.")
        return "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# Load the base model
def load_base_model(model_name, max_seq_length=2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
    return model, tokenizer

# Load dataset
def load_dataset_for_training():
    from datasets import load_dataset
    try:
        dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
        # Use a smaller subset for faster NAS iterations
        dataset = dataset.select(range(min(5000, len(dataset))))
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to small sample dataset")
        # Create a small sample dataset if the main one fails
        from datasets import Dataset
        sample_data = {
            "conversations": [
                [{"role": "user", "content": "Tell me about AI"}, 
                 {"role": "assistant", "content": "AI is a field of computer science..."}]
            ] * 10
        }
        return Dataset.from_dict(sample_data)

# Neural Architecture Search (NAS) for autonomous model improvement
class NeuralArchitectureSearch:
    def __init__(self, base_model_name, dataset, evaluation_metric="loss", checkpoint_path=None):
        self.base_model_name = base_model_name
        self.dataset = dataset
        self.evaluation_metric = evaluation_metric
        self.population = []
        self.best_architecture = None
        self.best_score = float('inf')  # Lower is better for loss
        self.generation_history = []
        self.checkpoint_path = os.path.join(SAVE_DIR, "nas_checkpoint.pkl") if checkpoint_path is None else checkpoint_path
        
    def initialize_population(self, population_size=5):
        """Initialize a population of model architectures with different configurations"""
        # Check if we can resume from checkpoint
        if os.path.exists(self.checkpoint_path):
            try:
                self.load_checkpoint()
                if len(self.population) > 0:
                    print(f"Resuming from checkpoint. Population size: {len(self.population)}")
                    return
                else:
                    print("Checkpoint found but population is empty. Creating new population.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Initializing fresh population...")
        
        print(f"Creating new population with size {population_size}")
        self.population = []  # Ensure we start with an empty population
        
        # Different LoRA configurations to try
        # Reduced size configurations for Colab's limited resources
        for i in range(population_size):
            # Sample random hyperparameters - keeping ranges smaller for Colab
            rank = 4 * torch.randint(low=1, high=17, size=(1,)).item()  # 4, 8, 12, ..., 64
            alpha = 8 * torch.randint(low=1, high=9, size=(1,)).item()  # 8, 16, 24, ..., 64
            dropout = torch.rand(1).item() * 0.2  # 0 to 0.2
            
            # Create configuration
            config = {
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                "use_rslora": bool(torch.randint(0, 2, (1,)).item()),  # Randomly use RSLoRA or not
            }
            
            self.population.append(config)
            print(f"Created configuration {i+1}: r={rank}, alpha={alpha}, dropout={dropout:.3f}, RSLoRA={config['use_rslora']}")
        
        # Create a safe initial best architecture (will be updated during evolution)
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        print(f"Initialized population with {len(self.population)} architectures")
        self.save_checkpoint()
    
    def evaluate_architecture(self, config, max_steps=20):
        """Train and evaluate a single architecture configuration"""
        print(f"Evaluating architecture: {config}")
        
        # Create model with this configuration
        try:
            # Clean memory before loading new model
            clean_gpu_memory()
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
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
            
            from transformers import TrainingArguments, DataCollatorForSeq2Seq
            from trl import SFTTrainer
            from unsloth import is_bfloat16_supported
            from unsloth.chat_templates import train_on_responses_only
            
            # Create a unique output dir for this evaluation
            config_id = f"r{config['r']}_a{config['lora_alpha']}_d{config['lora_dropout']:.2f}"
            output_dir = os.path.join(SAVE_DIR, f"eval_{config_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up trainer with smaller number of steps for quick evaluation
            eval_trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=self.dataset,
                dataset_text_field="conversations",
                max_seq_length=max_seq_length,
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
    
    def evolve_population(self, num_generations=3):
        """Evolve the population over multiple generations"""
        # Check if we have partial results from previous run
        start_gen = len(self.generation_history)
        
        # Initialize best architecture if none exists
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        for generation in range(start_gen, start_gen + num_generations):
            print(f"Generation {generation+1}/{start_gen + num_generations}")
            
            # Evaluate all architectures in the population
            scores = []
            for i, config in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{len(self.population)}")
                score = self.evaluate_architecture(config)
                scores.append((config, score))
                print(f"Architecture {i+1} score: {score}")
                
                # Update best architecture
                if score < self.best_score:
                    self.best_score = score
                    self.best_architecture = config.copy()
                    print(f"New best architecture found with score {score}")
                
                # Save checkpoint after each evaluation for resilience
                self.generation_history.append({"generation": generation, "scores": scores.copy()})
                self.save_checkpoint()
            
            # Sort architectures by score (lower is better)
            scores.sort(key=lambda x: x[1])
            
            if generation < start_gen + num_generations - 1:  # Don't evolve after last generation
                # Keep top 40% (elitism)
                elite_size = max(1, len(self.population) // 5)
                new_population = [config for config, _ in scores[:elite_size]]
                
                # Create new architectures through mutation and crossover
                while len(new_population) < len(self.population):
                    # Select parent(s)
                    if torch.rand(1).item() < 0.7:  # Crossover
                        parent1 = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0]
                        parent2 = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0]
                        
                        # Crossover
                        child = {}
                        for key in parent1:
                            if key == "target_modules":
                                child[key] = parent1[key]  # Keep this constant
                            else:
                                # Randomly select from either parent
                                child[key] = parent1[key] if torch.rand(1).item() < 0.5 else parent2[key]
                        
                        # Small mutation chance
                        if torch.rand(1).item() < 0.3:
                            if torch.rand(1).item() < 0.33:
                                # Mutate rank
                                child["r"] = max(4, child["r"] + torch.randint(-8, 9, (1,)).item() * 4)
                                # Make rank a multiple of 4
                                child["r"] = (child["r"] // 4) * 4
                                if child["r"] < 4:
                                    child["r"] = 4
                            elif torch.rand(1).item() < 0.5:
                                # Mutate alpha
                                child["lora_alpha"] = max(1, child["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8)
                            else:
                                # Mutate dropout
                                child["lora_dropout"] = max(0, min(0.2, child["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        new_population.append(child)
                    else:  # Mutation only
                        parent = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0].copy()
                        
                        # Apply mutations
                        mutation_strength = 0.5
                        if torch.rand(1).item() < mutation_strength:
                            parent["r"] = max(4, parent["r"] + torch.randint(-8, 9, (1,)).item() * 4)
                            # Make rank a multiple of 4
                            parent["r"] = (parent["r"] // 4) * 4
                            if parent["r"] < 4:
                                parent["r"] = 4
                        
                        if torch.rand(1).item() < mutation_strength:
                            parent["lora_alpha"] = max(1, parent["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8)
                        
                        if torch.rand(1).item() < mutation_strength:
                            parent["lora_dropout"] = max(0, min(0.2, parent["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        if torch.rand(1).item() < 0.2:  # 20% chance to flip RSLoRA
                            parent["use_rslora"] = not parent["use_rslora"]
                        
                        new_population.append(parent)
                
                self.population = new_population
                self.save_checkpoint()
        
        # Ensure best architecture exists before returning
        if self.best_architecture is None and len(self.population) > 0:
            # If no proper evaluation was made, just take the first configuration
            self.best_architecture = self.population[0].copy()
            print("Warning: No valid best architecture found during evolution. Using first configuration as default.")
        
        # Return best architecture found
        return self.best_architecture
    
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
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
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
            
            from transformers import TrainingArguments, DataCollatorForSeq2Seq
            from trl import SFTTrainer
            from unsloth import is_bfloat16_supported
            from unsloth.chat_templates import train_on_responses_only
            
            # Set up trainer with full training
            final_model_dir = os.path.join(SAVE_DIR, "nas_final_model")
            os.makedirs(final_model_dir, exist_ok=True)
            
            # Prepare dataset in the format expected by SFTTrainer
            try:
                # Debug original dataset format
                if len(self.dataset) > 0:
                    print(f"Original dataset format sample: {self.dataset[0]}")
                
                # Create a dataset with proper instruction-following examples
                from datasets import Dataset
                
                # Create sample instruction-response pairs for training
                # Since the original dataset has parsing issues, we'll create our own
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
                    }
                ] * 20  # Repeat to get enough training examples
                
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
                
                # Use the text_instruction_response format with SFTTrainer
                tokenizer = get_chat_template(
                    tokenizer, 
                    chat_template="llama-3.1"
                )
                
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
                
            final_trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=formatted_dataset,
                dataset_text_field="text",  # Changed to use the formatted text field
                max_seq_length=max_seq_length,
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
                    logging_steps=full_training_steps // 10,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=final_model_dir,
                    report_to="none",
                    # Colab specific settings
                    dataloader_num_workers=0,
                    disable_tqdm=False,
                    save_steps=full_training_steps // 3,  # Save checkpoints during training
                    save_total_limit=2,  # Keep only the most recent checkpoints
                ),
            )
            
            # Train the final model directly (no train_on_responses_only)
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
            
            # Save to Hugging Face Hub
            try:
                print("Saving model to Hugging Face Hub...")
                model.save_pretrained("CALISTA-INDUSTRY/NeuralArchitectureSearch")  # Local saving
                tokenizer.save_pretrained("CALISTA-INDUSTRY/NeuralArchitectureSearch")
                model.push_to_hub("CALISTA-INDUSTRY/NeuralArchitectureSearch", token = "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") # Online saving
                tokenizer.push_to_hub("CALISTA-INDUSTRY/NeuralArchitectureSearch", token = "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") # Online saving
                if True: model.push_to_hub_merged("CALISTA-INDUSTRY/NeuralArchitectureSearch", tokenizer, save_method="merged_16bit", token="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")
                if True: model.push_to_hub_gguf("CALISTA-INDUSTRY/NeuralArchitectureSearch", tokenizer, quantization_method = "f16", token = "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")
                print("Model successfully pushed to Hugging Face Hub")
            except Exception as e:
                print(f"Error pushing model to Hugging Face Hub: {e}")
                print("Continuing without pushing to Hub...")
            
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
    
    def save_checkpoint(self):
        """Save checkpoint to resume training later"""
        checkpoint = {
            "population": self.population,
            "best_architecture": self.best_architecture,
            "best_score": self.best_score,
            "generation_history": self.generation_history
        }
        
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"Checkpoint saved to {self.checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self):
        """Load checkpoint to resume training"""
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.population = checkpoint["population"]
            self.best_architecture = checkpoint["best_architecture"]
            self.best_score = checkpoint["best_score"]
            self.generation_history = checkpoint.get("generation_history", [])
            
            print(f"Checkpoint loaded from {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

def run_nas_optimization():
    """Main function to run the Neural Architecture Search optimization"""
    print("Starting Neural Architecture Search process...")
    
    # Get appropriate model based on available GPU
    model_name = get_appropriate_model_name()
    print(f"Using model: {model_name}")
    
    # Load dataset
    dataset = load_dataset_for_training()
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Initialize NAS
    nas = NeuralArchitectureSearch(
        base_model_name=model_name,
        dataset=dataset
    )
    
    # Run with smaller population and generations for Colab
    population_size = 4  # Smaller population for Colab
    num_generations = 2  # Fewer generations for Colab
    
    # Initialize population and run evolution
    nas.initialize_population(population_size=population_size)
    best_config = nas.evolve_population(num_generations=num_generations)
    
    # Check if a valid best configuration was found
    if best_config is None:
        print("No valid architecture configuration was found. Using default configuration.")
        # Create a default configuration as fallback
        best_config = {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "use_rslora": False,
        }
        nas.best_architecture = best_config
        nas.best_score = float('inf')
        nas.save_checkpoint()
    
    print(f"Best architecture found: {best_config}")
    print(f"Best evaluation score: {nas.best_score}")
    
    # Train the final model with the best architecture
    # Use fewer steps for Colab to avoid timeouts
    final_model, final_tokenizer, train_results = nas.train_final_model(best_config, full_training_steps=40)
    
    if train_results:
        print("Neural Architecture Search completed successfully!")
        print(f"Final training loss: {train_results.training_loss}")
        print(f"Model saved to {os.path.join(SAVE_DIR, 'nas_final_model')}")
    else:
        print("Neural Architecture Search completed but final training failed.")
        print(f"Best architecture configuration saved to {os.path.join(SAVE_DIR, 'nas_checkpoint.pkl')}")

if __name__ == "__main__":
    run_nas_optimization()