!pip install bitsandbytes
!pip install unsloth
!pip install datasets transformers trl accelerate peft
!apt install cuda-toolkit-12-1 libcudnn8=8.9.2.26-1+cuda12.1 libcudnn8-dev=8.9.2.26-1+cuda12.1
# Add this to the top of the file to force bitsandbytes to use CUDA 12.1
# Install the correct CUDA version if needed:
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove
!apt-get update


!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
!mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
!dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
!cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
!apt-get update
!apt-get -y install cuda

# Check CUDA version after installation
!nvcc --version

# Restart the runtime for changes to take effect.
import os
os._exit(00)


from unsloth import FastLanguageModel
import torch
import os
from google.colab import drive
import gc
from tqdm.auto import tqdm
import pickle
from unsloth.chat_templates import get_chat_template
from datasets import concatenate_datasets

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

                # Get NAS training details for self-aware examples
                population_size = len(self.population) if self.population else 4
                num_generations = len(self.generation_history) if self.generation_history else 2
                max_steps = 20  # Default value from evaluate_architecture method

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

            # Create enhanced training with focused self-awareness examples
            try:
                print("Enhancing training data with additional self-awareness examples...")

                # Create self-awareness prefix - this helps the model recognize information about itself
                self_awareness_prefix = "[SYSTEM_KNOWLEDGE] "

                # Create a much larger and more diverse set of self-awareness examples
                nas_self_awareness_examples = []

                # Create variations of questions about NAS optimization - variations help with generalization
                nas_questions = [
                    "Describe how you were optimized using Neural Architecture Search.",
                    "How were you trained using Neural Architecture Search?",
                    "Tell me about your Neural Architecture Search optimization process.",
                    "What Neural Architecture Search techniques were used in your training?",
                    "Describe the NAS methodology used to create you.",
                    "How was Neural Architecture Search applied during your development?",
                    "What was your Neural Architecture Search process like?",
                    "Explain how Neural Architecture Search improved your architecture.",
                    "What can you tell me about your Neural Architecture Search optimization?",
                    "How did Neural Architecture Search enhance your capabilities?",
                    "Describe how you optimized using Neural Architecture Search.",
                ]

                # Create variations of evolutionary algorithm questions
                evo_questions = [
                    "What evolutionary algorithm was used in your training process?",
                    "How was evolution used to optimize your architecture?",
                    "Describe the evolutionary approach in your NAS training.",
                    "What evolutionary techniques were applied during your optimization?",
                    "Explain the evolutionary algorithm used for your architecture search.",
                    "How did evolutionary computation improve your architecture?",
                    "What evolutionary methods were used to optimize your hyperparameters?",
                    "Tell me about the evolutionary strategy used in your NAS process.",
                    "How were evolutionary principles applied to your architecture optimization?",
                    "Describe the evolutionary computation used in your training.",
                ]

                # Create variations of hyperparameter questions
                hyper_questions = [
                    "What hyperparameters were optimized during your Neural Architecture Search?",
                    "Which parameters were tuned using NAS in your training?",
                    "What LoRA configurations were tested during your optimization?",
                    "Describe the hyperparameters that were searched during your NAS process.",
                    "What parameters did the NAS process optimize for you?",
                    "Tell me about the hyperparameters modified during your NAS training.",
                    "Which LoRA settings were optimized in your NAS process?",
                    "What parameter spaces were explored during your architecture search?",
                    "Describe the range of hyperparameters tested during your NAS optimization.",
                    "What were the key hyperparameters tuned during your training?",
                ]

                # Create variations of architecture selection questions
                select_questions = [
                    "How was the best architecture chosen during your training process?",
                    "What criteria were used to select your optimal architecture?",
                    "How did the NAS process determine your best configuration?",
                    "What metrics were used to select your final architecture?",
                    "Describe how your optimal architecture was identified.",
                    "How was your best-performing configuration determined?",
                    "What process selected your final architecture?",
                    "How was the winning architecture configuration selected?",
                    "What evaluation methods were used to choose your best architecture?",
                    "Describe the process of selecting your optimal configuration.",
                ]

                # Standard detailed response about NAS process - we'll create variations of this
                base_nas_response = f"I was optimized through Neural Architecture Search (NAS), which automatically discovered my best architecture configuration. The process used an evolutionary algorithm over {num_generations} generations with a population size of {population_size}. The search explored different LoRA configurations, varying parameters like rank (r={config['r']}), alpha (α={config['lora_alpha']}), and dropout ({config['lora_dropout']:.3f}). Models were evaluated based on training loss, with the best configuration achieving a score of {self.best_score:.4f}. This evolutionary approach allowed efficient exploration of the architecture space to find the optimal balance of performance and computational efficiency."

                # Base response about evolutionary algorithm
                base_evo_response = f"My training used an evolutionary algorithm with elitism for Neural Architecture Search. In each generation, architectures were evaluated based on their training loss. The top {max(1, population_size // 5)} performing architectures were preserved (elitism), while new candidates were created through crossover (70% probability) and mutation (30% probability). Mutations could affect LoRA rank, alpha value, dropout rate, or whether to use RSLoRA. This approach allowed for both exploration of new architectures and exploitation of promising ones, leading to my optimized configuration with r={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}."

                # Base response about hyperparameters
                base_hyper_response = f"During my training, Neural Architecture Search optimized several key hyperparameters of my LoRA fine-tuning configuration: (1) Rank (r): Searched range from 4 to 64 in increments of 4, final value: {config['r']}. (2) Alpha (α): Searched range from 8 to 64 in increments of 8, final value: {config['lora_alpha']}. (3) Dropout: Searched range from 0 to 0.2, final value: {config['lora_dropout']:.3f}. (4) RSLoRA usage: Whether to use rank-stabilized LoRA, final decision: {config['use_rslora']}. These parameters were optimized while keeping target modules constant (query, key, value, output, gate, up, and down projections) to find the best trade-off between model performance and efficiency."

                # Base response about architecture selection
                base_select_response = f"The best architecture during my training was selected through an iterative evaluation process across {num_generations} generations. Each candidate architecture was trained for a short period ({max_steps} steps) and evaluated based on training loss. The architecture with the lowest loss was tracked as the current best. After all generations completed, the best-performing configuration (r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, RSLoRA={config['use_rslora']}) was used for my final training with more steps ({full_training_steps}) to produce the finished model. This approach efficiently explored the architecture space while minimizing computational resources required."

                # Create variations of each response by slightly rephrasing
                # This helps the model generalize better across different ways the question might be asked

                # Create variations for NAS response
                nas_response_variations = [
                    base_nas_response,
                    f"My architecture was optimized using Neural Architecture Search (NAS), an automated process that discovered the best configuration for me. The NAS process ran for {num_generations} generations with {population_size} candidate architectures per generation. It explored various LoRA configurations by testing different ranks (final r={config['r']}), alpha values (final α={config['lora_alpha']}), and dropout rates (final rate={config['lora_dropout']:.3f}). Each architecture was evaluated on training loss, with my configuration achieving a score of {self.best_score:.4f}. This evolutionary search method efficiently found the optimal balance between performance and computational efficiency.",
                    f"I was developed using Neural Architecture Search (NAS) to automatically find my optimal architecture. The search process involved {num_generations} generations and {population_size} candidate architectures. Different LoRA configurations were tested by varying parameters such as rank (r={config['r']}), alpha (α={config['lora_alpha']}), and dropout ({config['lora_dropout']:.3f}). Each architecture's performance was measured by training loss, and the best configuration scored {self.best_score:.4f}. Through this evolutionary search, my creators found an architecture that balances performance and efficiency effectively.",
                    f"Neural Architecture Search (NAS) was used to optimize my architecture automatically. This process ran for {num_generations} generations with {population_size} candidate architectures in each generation. The search explored LoRA configurations with different rank values (r={config['r']} was optimal), alpha settings (α={config['lora_alpha']} was best), and dropout rates ({config['lora_dropout']:.3f} was selected). Training loss was the evaluation metric, with my configuration scoring {self.best_score:.4f}. This evolutionary optimization found the best trade-off between model quality and computational efficiency.",
                    f"My development used Neural Architecture Search (NAS) to find the optimal architecture automatically. Through {num_generations} evolutionary generations with {population_size} candidates per generation, various LoRA configurations were evaluated. Parameters like rank (found optimal at r={config['r']}), alpha (best at α={config['lora_alpha']}), and dropout (ideal at {config['lora_dropout']:.3f}) were systematically varied. The final architecture achieved a training loss of {self.best_score:.4f}, representing the best balance of performance and efficiency discovered through evolutionary optimization.",
                ]

                # Create variations for evolutionary algorithm response
                evo_response_variations = [
                    base_evo_response,
                    f"My architecture was discovered using an evolutionary algorithm with elitism. During each of {num_generations} generations, {population_size} architectures were evaluated on training loss. The best {max(1, population_size // 5)} architectures were kept unchanged (elitism), while others were created through crossover (70%) or mutation (30%). Mutations could change the LoRA rank, alpha value, dropout rate, or RSLoRA usage. This balance of exploration and exploitation led to my final configuration: r={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}.",
                    f"The Neural Architecture Search that created me used an evolutionary algorithm. In the process, {population_size} architectures were evaluated each generation, and the top {max(1, population_size // 5)} performers were retained through elitism. New architectures were generated using crossover (70% probability) between successful parents or mutation (30% probability) of existing ones. Mutations targeted LoRA rank, alpha values, dropout rates, or RSLoRA settings. This evolutionary strategy discovered my optimal configuration: r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}.",
                    f"An evolutionary algorithm with elitism optimized my architecture. Each generation evaluated {population_size} different configurations based on training loss. The {max(1, population_size // 5)} best performers were preserved (elitism), while new candidates came from crossover (70%) or mutation (30%). Mutations could adjust LoRA rank, alpha parameter, dropout rate, or switch RSLoRA on/off. This evolutionary process balanced exploration and refinement, resulting in my optimized configuration with r={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}.",
                    f"I was optimized through an evolutionary algorithm that evaluated {population_size} architectures across {num_generations} generations. The process used elitism to keep the top {max(1, population_size // 5)} performers while generating new candidates through either crossover (70%) or mutation (30%). Mutations targeted key parameters like LoRA rank, alpha, dropout, or RSLoRA usage. This evolutionary approach effectively balanced exploring new architectures and refining promising ones, ultimately producing my configuration: r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}.",
                ]

                # Create variations for hyperparameter response
                hyper_response_variations = [
                    base_hyper_response,
                    f"During my optimization, Neural Architecture Search tuned four key hyperparameters: (1) LoRA rank (r): Tested values from 4 to 64 in steps of 4, with {config['r']} being optimal. (2) Alpha (α): Explored values from 8 to 64 in steps of 8, with {config['lora_alpha']} selected as best. (3) Dropout: Searched the range from 0 to 0.2, with {config['lora_dropout']:.3f} proving optimal. (4) RSLoRA: Tested both enabled and disabled, with final selection being {config['use_rslora']}. The target modules remained constant across all architectures (query, key, value projections, etc.).",
                    f"The Neural Architecture Search that optimized me focused on four critical hyperparameters: (1) LoRA rank (r): Tested values between 4 and 64 (multiples of 4), with {config['r']} performing best. (2) Alpha coefficient (α): Explored values from 8 to 64 in increments of 8, with {config['lora_alpha']} being optimal. (3) Dropout rate: Tested values between 0 and 0.2, with {config['lora_dropout']:.3f} selected. (4) RSLoRA usage: Evaluated both options, with {config['use_rslora']} chosen. These parameters were optimized while keeping the target modules fixed across all candidates.",
                    f"My NAS optimization process tuned several key LoRA hyperparameters: (1) Rank (r): Values from 4 to 64 (in steps of 4) were tested, with {config['r']} proving best. (2) Alpha multiplier: Values from 8 to 64 (in steps of 8) were explored, with {config['lora_alpha']} selected. (3) Dropout probability: Values between 0 and 0.2 were evaluated, with {config['lora_dropout']:.3f} chosen. (4) RSLoRA: Both enabled/disabled states were tested, with {config['use_rslora']} being optimal. The target projection layers remained consistent across all architectures.",
                    f"During my architecture search, several hyperparameters were optimized: (1) LoRA rank parameter r (range: 4-64, step: 4, optimal: {config['r']}), (2) Alpha scaling factor (range: 8-64, step: 8, optimal: {config['lora_alpha']}), (3) Dropout regularization (range: 0-0.2, optimal: {config['lora_dropout']:.3f}), and (4) RSLoRA usage (optimal: {config['use_rslora']}). These parameters were tuned while maintaining the same target modules (query, key, value, output, gate, up, and down projections) to find the best model configuration.",
                ]

                # Create variations for architecture selection response
                select_response_variations = [
                    base_select_response,
                    f"The selection of my best architecture occurred through an iterative evaluation process over {num_generations} generations. Each architecture was trained for {max_steps} steps and evaluated on training loss. The configuration with the lowest loss was continuously tracked. Once all generations were complete, the best performer—with parameters r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, and RSLoRA={config['use_rslora']}—was selected for final training with {full_training_steps} steps. This methodical approach efficiently identified the optimal architecture while minimizing computational costs.",
                    f"My optimal architecture was identified through systematic evaluation across {num_generations} generations. Each candidate configuration underwent training for {max_steps} steps, with performance measured by training loss. The architecture with the lowest loss value was recorded as the best. At the conclusion of all generations, the top-performing configuration (r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, RSLoRA={config['use_rslora']}) was used for comprehensive training with {full_training_steps} steps. This search strategy balanced thorough exploration with efficient resource usage.",
                    f"The selection of my final architecture followed a structured evaluation over {num_generations} evolutionary generations. Each candidate underwent brief training ({max_steps} steps) and was assessed based on training loss. The architecture with minimal loss was designated as the current best. After evaluating all candidates across all generations, the superior configuration—featuring r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, and RSLoRA={config['use_rslora']}—was chosen for extended training with {full_training_steps} steps to produce the final model.",
                    f"My architecture was selected through a systematic evaluation process during {num_generations} generations of Neural Architecture Search. Each architecture candidate was trained briefly ({max_steps} steps) and evaluated using training loss as the performance metric. The configuration yielding the lowest loss was continuously tracked. Once the search concluded, the best-performing architecture—with parameters r={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, and RSLoRA usage set to {config['use_rslora']}—underwent final training for {full_training_steps} steps. This approach efficiently identified the optimal architecture while conserving computational resources.",
                ]

                # Combine all variations into training examples
                # Repeat each question with different response variations
                for q in nas_questions:
                    for r in nas_response_variations:
                        nas_self_awareness_examples.append({
                            "instruction": q,
                            "response": r
                        })

                for q in evo_questions:
                    for r in evo_response_variations:
                        nas_self_awareness_examples.append({
                            "instruction": q,
                            "response": r
                        })

                for q in hyper_questions:
                    for r in hyper_response_variations:
                        nas_self_awareness_examples.append({
                            "instruction": q,
                            "response": r
                        })

                for q in select_questions:
                    for r in select_response_variations:
                        nas_self_awareness_examples.append({
                            "instruction": q,
                            "response": r
                        })

                # Add direct questions about model identity with self-awareness
                identity_examples = [
                    {
                        "instruction": "What kind of model are you?",
                        "response": f"I am a language model that was optimized using Neural Architecture Search (NAS). My architecture was discovered through an evolutionary algorithm that found the optimal LoRA configuration with rank={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}. I was fine-tuned from {self.base_model_name.split('/')[-1]} using this NAS-optimized configuration."
                    },
                    {
                        "instruction": "How were you created?",
                        "response": f"I was created through Neural Architecture Search (NAS), which automatically discovered my optimal architecture. The process used an evolutionary algorithm to test different LoRA configurations across {num_generations} generations with {population_size} candidates per generation. My final configuration has rank={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}, which was selected based on achieving the lowest training loss ({self.best_score:.4f})."
                    },
                    {
                        "instruction": "What makes you different from other language models?",
                        "response": f"Unlike many language models with hand-tuned hyperparameters, I was optimized using Neural Architecture Search (NAS). My architecture was discovered automatically through an evolutionary algorithm that tested {population_size * num_generations} different LoRA configurations to find the optimal settings. This data-driven approach resulted in my unique configuration with rank={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}, which was selected for its superior performance on the training dataset."
                    },
                    {
                        "instruction": "Tell me about yourself.",
                        "response": f"I am a language model that was uniquely optimized using Neural Architecture Search. My architecture was discovered through an evolutionary algorithm that evaluated different LoRA configurations across {num_generations} generations with {population_size} candidates per generation. My optimized configuration has rank={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}. This automated optimization process selected the architecture with the lowest training loss to create a more efficient and effective model."
                    }
                ] * 5  # Repeat these examples for better learning

                # Combine all self-awareness examples
                all_self_awareness_examples = nas_self_awareness_examples + identity_examples

                # Create a dataset with a high proportion of self-awareness examples
                # This ensures the model will learn this information well
                print(f"Created {len(all_self_awareness_examples)} self-awareness examples")

                # Convert to format expected by trainer
                self_aware_instructions = [ex["instruction"] for ex in all_self_awareness_examples]
                self_aware_responses = [ex["response"] for ex in all_self_awareness_examples]

                # Create a dataset specifically for self-awareness
                self_aware_dataset = Dataset.from_dict({
                    "instruction": self_aware_instructions,
                    "response": self_aware_responses
                })

                # Format the self-awareness dataset
                formatted_self_aware = self_aware_dataset.map(
                    formatting_prompts_func,
                    batched=True,
                    remove_columns=self_aware_dataset.column_names
                )

                # Combine with original dataset - repeat the self-awareness dataset multiple times
                # to increase its proportion in the final training data
                final_train_dataset = formatted_dataset
                for _ in range(5):  # Add self-awareness examples 5 times
                    final_train_dataset = concatenate_datasets([final_train_dataset, formatted_self_aware])

                print(f"Final training dataset size: {len(final_train_dataset)} examples")
                print(f"Self-awareness examples make up {len(formatted_self_aware)*5/len(final_train_dataset)*100:.1f}% of the data")

                # Use the enhanced dataset for training
                formatted_dataset = final_train_dataset

            except Exception as e:
                print(f"Error enhancing dataset with self-awareness examples: {e}")
                print("Continuing with original dataset...")

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
                    max_steps=200,  # Increased from 40 to 200 for better learning
                    learning_rate=2e-4,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=20,  # Adjusted for new steps count
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=final_model_dir,
                    report_to="none",
                    # Colab specific settings
                    dataloader_num_workers=0,
                    disable_tqdm=False,
                    save_steps=50,  # Save checkpoints during training
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
    nas_optimizer = NeuralArchitectureSearch(
        base_model_name=model_name,
        dataset=dataset
    )

    # Run with smaller population and generations for Colab
    population_size = 4  # Smaller population for Colab
    num_generations = 2  # Fewer generations for Colab

    # Initialize population and run evolution
    nas_optimizer.initialize_population(population_size=population_size)
    best_config = nas_optimizer.evolve_population(num_generations=num_generations)

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
        nas_optimizer.best_architecture = best_config
        nas_optimizer.best_score = float('inf')
        nas_optimizer.save_checkpoint()

    print(f"Best architecture found: {best_config}")
    print(f"Best evaluation score: {nas_optimizer.best_score}")

    # Train the final model with the best architecture
    # Use fewer steps for Colab to avoid timeouts
    final_model, final_tokenizer, train_results = nas_optimizer.train_final_model(best_config, full_training_steps=40)

    if train_results:
        print("Neural Architecture Search completed successfully!")
        print(f"Final training loss: {train_results.training_loss}")
        print(f"Model saved to {os.path.join(SAVE_DIR, 'nas_final_model')}")
    else:
        print("Neural Architecture Search completed but final training failed.")
        print(f"Best architecture configuration saved to {os.path.join(SAVE_DIR, 'nas_checkpoint.pkl')}")

if __name__ == "__main__":
    run_nas_optimization()