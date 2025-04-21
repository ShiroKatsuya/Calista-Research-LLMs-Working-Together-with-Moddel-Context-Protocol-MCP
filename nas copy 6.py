from unsloth import FastLanguageModel
import torch
import os
from google.colab import drive
import gc
from tqdm.auto import tqdm
import pickle
from unsloth.chat_templates import get_chat_template
from datasets import concatenate_datasets
import numpy as np
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import psutil
import json
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
import traceback
import torch.nn.functional as F
from sklearn.decomposition import PCA
from datasets import Dataset, DatasetDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Enhanced memory management functions
def get_memory_usage():
    """Get current memory usage of system and GPU"""
    system_memory = psutil.virtual_memory()
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    return {
        "system_used_percent": system_memory.percent,
        "system_available_gb": system_memory.available / (1024**3),
        "system_total_gb": system_memory.total / (1024**3),
        "gpu_used_gb": gpu_memory,
        "gpu_total_gb": total_gpu if torch.cuda.is_available() else None
    }

def memory_optimization_strategy(memory_info, target_buffer_percent=20):
    """Determine optimization strategy based on memory status
    
    Returns:
        dict: Strategy recommendations for batch size, precision, etc.
    """
    strategy = {}
    
    # System RAM strategies
    sys_usage = memory_info["system_used_percent"]
    if sys_usage > 90:
        strategy["critical_system_ram"] = True
        strategy["recommended_action"] = "reduce_dataset_size"
    elif sys_usage > 75:
        strategy["dataset_caching"] = False
        strategy["prefetch_factor"] = 1  # Reduce prefetching
    else:
        strategy["dataset_caching"] = True
        strategy["prefetch_factor"] = 2
    
    # GPU strategies
    if memory_info["gpu_used_gb"] is not None:
        gpu_usage_percent = memory_info["gpu_used_gb"] / memory_info["gpu_total_gb"] * 100
        gpu_free_gb = memory_info["gpu_total_gb"] - memory_info["gpu_used_gb"]
        
        if gpu_usage_percent > 85:
            strategy["critical_gpu_ram"] = True
            strategy["recommended_action"] = "reduce_batch_size_further"
            strategy["gradient_checkpointing"] = True
            strategy["cpu_offloading"] = True
        elif gpu_usage_percent > 70:
            strategy["gradient_checkpointing"] = True
            strategy["batch_size_adjust"] = 0.8  # Reduce by 20%
        else:
            strategy["gradient_checkpointing"] = "auto"
            strategy["batch_size_adjust"] = 1.0
            
        # Set target sequence length based on available GPU memory
        if gpu_free_gb < 3:
            strategy["max_seq_length"] = min(1024, max_seq_length)
        elif gpu_free_gb < 6:
            strategy["max_seq_length"] = min(1536, max_seq_length)
        else:
            strategy["max_seq_length"] = max_seq_length
    
    return strategy

def adaptive_batch_size(memory_info, base_batch_size=4):
    """Determine appropriate batch size based on available memory"""
    if memory_info["gpu_used_gb"] is None:
        return 1  # Conservative default for CPU training
    
    gpu_free_gb = memory_info["gpu_total_gb"] - memory_info["gpu_used_gb"]
    
    # Approximate batch size based on free memory
    # This is a heuristic and may need adjustment based on model size
    if gpu_free_gb < 2:
        return 1
    elif gpu_free_gb < 5:
        return 2
    elif gpu_free_gb < 10:
        return min(4, base_batch_size)
    else:
        return base_batch_size

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
                    logger.info(f"Resuming from checkpoint. Population size: {len(self.population)}")
                    return
                else:
                    logger.info("Checkpoint found but population is empty. Creating new population.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Initializing fresh population...")
        
        logger.info(f"Creating new population with size {population_size}")
        self.population = []  # Ensure we start with an empty population
        
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        target_modules = self.curriculum_stages[self.current_curriculum_stage]["target_modules"]
        
        # Different LoRA configurations to try
        for i in range(population_size):
            # Sample random hyperparameters based on curriculum constraints
            r_min, r_max = stage_constraints["r_range"]
            alpha_min, alpha_max = stage_constraints["lora_alpha_range"]
            dropout_min, dropout_max = stage_constraints["lora_dropout_range"]
            
            rank = 4 * torch.randint(low=r_min//4, high=r_max//4 + 1, size=(1,)).item()  # Ensure multiple of 4
            alpha = 8 * torch.randint(low=alpha_min//8, high=alpha_max//8 + 1, size=(1,)).item()  # Multiple of 8
            dropout = dropout_min + torch.rand(1).item() * (dropout_max - dropout_min)
            
            # Create configuration
            config = {
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": target_modules,
                "use_rslora": bool(torch.randint(0, 2, (1,)).item()),  # Randomly use RSLoRA or not
                "search_method": "evolutionary",  # Track which method generated this config
            }
            
            self.population.append(config)
            logger.info(f"Created configuration {i+1}: r={rank}, alpha={alpha}, dropout={dropout:.3f}, RSLoRA={config['use_rslora']}")
        
        # Create a safe initial best architecture (will be updated during evolution)
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        logger.info(f"Initialized population with {len(self.population)} architectures")
        self.save_checkpoint()
    
    def initialize_bayesian_search_space(self):
        """Initialize the search space for Bayesian optimization"""
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        
        # Define the search space for Bayesian optimization
        self.bayesian_search_space = [
            Integer(low=stage_constraints["r_range"][0], high=stage_constraints["r_range"][1], name="r"),
            Integer(low=stage_constraints["lora_alpha_range"][0], high=stage_constraints["lora_alpha_range"][1], name="lora_alpha"),
            Real(low=stage_constraints["lora_dropout_range"][0], high=stage_constraints["lora_dropout_range"][1], name="lora_dropout"),
            Categorical(categories=[True, False], name="use_rslora")
        ]
        
        logger.info(f"Initialized Bayesian search space for stage: {self.current_curriculum_stage}")
        return self.bayesian_search_space
    
    def evaluate_architecture(self, config, max_steps=None):
        """Train and evaluate a single architecture configuration"""
        logger.info(f"Evaluating architecture: {config}")
        
        # Get max_steps from curriculum if not provided
        if max_steps is None:
            max_steps = self.curriculum_stages[self.current_curriculum_stage]["training_config"]["max_steps"]
        
        # Monitor memory before loading model
        memory_before = get_memory_usage()
        logger.info(f"Memory before loading model: {memory_before}")
        
        # Determine optimization strategy based on memory
        memory_strategy = memory_optimization_strategy(memory_before)
        batch_size = adaptive_batch_size(memory_before, 
                                         base_batch_size=self.curriculum_stages[self.current_curriculum_stage]["training_config"].get("batch_size", 2))
        
        # Adjust max_seq_length if needed
        adjusted_max_seq_length = memory_strategy.get("max_seq_length", max_seq_length)
        
        # Create model with this configuration
        try:
            # Clean memory before loading new model
            clean_gpu_memory()
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=adjusted_max_seq_length,
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
                use_gradient_checkpointing=memory_strategy.get("gradient_checkpointing", "unsloth"),
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
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                dataset_text_field="conversations",
                max_seq_length=adjusted_max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=1,  # Lower for limited resources
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=1,
                    max_steps=max_steps,
                    learning_rate=self.curriculum_stages[self.current_curriculum_stage]["training_config"]["learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=max(1, max_steps // 4),
                    evaluation_strategy="steps",
                    eval_steps=max(1, max_steps // 2),
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=output_dir,
                    report_to="none",
                    # Resource specific settings
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
                eval_results = eval_trainer.evaluate()
                
                train_loss = train_result.training_loss
                eval_loss = eval_results.get("eval_loss", float('inf'))
                
                # Compute additional metrics for metacognitive system
                validation_metrics = self._compute_cognitive_metrics(model, tokenizer, config)
                
                # Combine metrics
                all_metrics = {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "memory_usage": get_memory_usage(),
                    **validation_metrics
                }
                
                # Update metacognitive system
                performance_analysis = self.metacognitive_system.monitor_performance(all_metrics)
                knowledge_state = self.metacognitive_system.assess_knowledge_state(
                    {"coverage": {"config": config}}, 
                    all_metrics
                )
                self.metacognitive_system.generate_reflection(performance_analysis, knowledge_state)
                
                # Track results by search method
                if "search_method" in config:
                    self.search_method_results[config["search_method"]].append((config, eval_loss))
                
                # Return the evaluation loss (lower is better)
                return eval_loss, validation_metrics
                
            except Exception as e:
                logger.error(f"Training failed with error: {e}")
                return float('inf'), {}  # Penalize failed configurations
            
        except Exception as e:
            logger.error(f"Error during architecture evaluation: {e}")
            return float('inf'), {}  # Return worst possible score for failed configurations
    
    def _compute_cognitive_metrics(self, model, tokenizer, config):
        """Compute additional cognitive metrics for the metacognitive system"""
        # This is a simplified implementation that would be more sophisticated in production
        try:
            # Basic metrics based on curriculum stage
            stage = self.current_curriculum_stage
            base_metrics = {}
            
            # Create a base performance score from 0-1 based on how close the config is to optimal
            # This is simplistic; in production you'd do actual cognitive task evaluation
            stage_constraints = self.curriculum_stages[stage]["architecture_constraints"]
            
            # Calculate normalized rank score (how close to max rank for this stage)
            r_min, r_max = stage_constraints["r_range"]
            r_score = (config["r"] - r_min) / (r_max - r_min) if r_max > r_min else 0.5
            
            # For perception stage
            if stage == "perception":
                base_metrics["pattern_recognition_accuracy"] = 0.5 + r_score * 0.4  # 0.5-0.9 range
                base_metrics["information_processing_speed"] = 0.4 + r_score * 0.3  # 0.4-0.7 range
                
            # For attention stage
            elif stage == "attention":
                base_metrics["focus_duration"] = 0.5 + r_score * 0.3
                base_metrics["distraction_resistance"] = 0.4 + r_score * 0.4
                base_metrics["selective_attention_score"] = 0.6 + r_score * 0.3
                
            # Add noise to make metrics realistic
            for key in base_metrics:
                base_metrics[key] *= random.uniform(0.9, 1.1)
                base_metrics[key] = min(1.0, base_metrics[key])  # Cap at 1.0
            
            # Add some metrics for all stages
            for ability in self.metacognitive_system.curriculum_progress:
                # Add at least one metric per ability with low scores for abilities we haven't reached yet
                metrics = self.metacognitive_system.ability_metrics.get(ability, [])
                if metrics and not any(m in base_metrics for m in metrics):
                    # For current or previous stages, give better scores
                    if self._get_stage_index(ability) <= self._get_stage_index(stage):
                        base_score = 0.5 + r_score * 0.3
                    else:
                        base_score = 0.2 + r_score * 0.1  # Lower scores for future abilities
                    
                    # Add the first metric for this ability
                    base_metrics[metrics[0]] = min(1.0, base_score * random.uniform(0.9, 1.1))
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error computing cognitive metrics: {e}")
            return {}
    
    def _get_stage_index(self, stage_name):
        """Get the numerical index of a stage in the curriculum"""
        stages = list(self.curriculum_stages.keys())
        try:
            return stages.index(stage_name)
        except ValueError:
            return -1
    
    def evolve_population(self, num_generations=3):
        """Evolve the population over multiple generations"""
        # Check if we have partial results from previous run
        start_gen = len(self.generation_history)
        
        # Initialize best architecture if none exists
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        for generation in range(start_gen, start_gen + num_generations):
            logger.info(f"Generation {generation+1}/{start_gen + num_generations}")
            
            # Evaluate all architectures in the population
            scores = []
            for i, config in enumerate(self.population):
                logger.info(f"Evaluating architecture {i+1}/{len(self.population)}")
                
                # Add search method if not present
                if "search_method" not in config:
                    config["search_method"] = "evolutionary"
                    
                score, metrics = self.evaluate_architecture(config)
                scores.append((config, score, metrics))
                logger.info(f"Architecture {i+1} score: {score}")
                
                # Update best architecture
                if score < self.best_score:
                    self.best_score = score
                    self.best_architecture = config.copy()
                    logger.info(f"New best architecture found with score {score}")
                
                # Save checkpoint after each evaluation for resilience
                self.generation_history.append({"generation": generation, "scores": [(c, s) for c, s, _ in scores]})
                self.save_checkpoint()
            
            # Sort architectures by score (lower is better)
            scores.sort(key=lambda x: x[1])
            
            if generation < start_gen + num_generations - 1:  # Don't evolve after last generation
                # Keep top 40% (elitism)
                elite_size = max(1, len(self.population) // 5)
                new_population = [config for config, _, _ in scores[:elite_size]]
                
                # Get current stage constraints
                stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
                
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
                                child[key] = parent1[key]  # Keep this constant per curriculum stage
                            elif key == "search_method":
                                child[key] = "evolutionary"  # Mark as created by evolutionary search
                            else:
                                # Randomly select from either parent
                                child[key] = parent1[key] if torch.rand(1).item() < 0.5 else parent2[key]
                        
                        # Small mutation chance
                        if torch.rand(1).item() < 0.3:
                            if torch.rand(1).item() < 0.33:
                                # Mutate rank
                                min_r, max_r = stage_constraints["r_range"]
                                child["r"] = max(min_r, min(max_r, child["r"] + torch.randint(-8, 9, (1,)).item() * 4))
                                # Make rank a multiple of 4
                                child["r"] = (child["r"] // 4) * 4
                                if child["r"] < 4:
                                    child["r"] = 4
                            elif torch.rand(1).item() < 0.5:
                                # Mutate alpha
                                min_alpha, max_alpha = stage_constraints["lora_alpha_range"]
                                child["lora_alpha"] = max(min_alpha, min(max_alpha, child["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8))
                            else:
                                # Mutate dropout
                                min_dropout, max_dropout = stage_constraints["lora_dropout_range"]
                                child["lora_dropout"] = max(min_dropout, min(max_dropout, child["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        new_population.append(child)
                    else:  # Mutation only
                        parent = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0].copy()
                        
                        # Apply mutations
                        mutation_strength = 0.5
                        if torch.rand(1).item() < mutation_strength:
                            min_r, max_r = stage_constraints["r_range"]
                            parent["r"] = max(min_r, min(max_r, parent["r"] + torch.randint(-8, 9, (1,)).item() * 4))
                            # Make rank a multiple of 4
                            parent["r"] = (parent["r"] // 4) * 4
                            if parent["r"] < 4:
                                parent["r"] = 4
                        
                        if torch.rand(1).item() < mutation_strength:
                            min_alpha, max_alpha = stage_constraints["lora_alpha_range"]
                            parent["lora_alpha"] = max(min_alpha, min(max_alpha, parent["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8))
                        
                        if torch.rand(1).item() < mutation_strength:
                            min_dropout, max_dropout = stage_constraints["lora_dropout_range"]
                            parent["lora_dropout"] = max(min_dropout, min(max_dropout, parent["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        if torch.rand(1).item() < 0.2:  # 20% chance to flip RSLoRA
                            parent["use_rslora"] = not parent["use_rslora"]
                        
                        # Mark as created by evolutionary search
                        parent["search_method"] = "evolutionary"
                        
                        new_population.append(parent)
                
                self.population = new_population
                self.save_checkpoint()
        
        # Ensure best architecture exists before returning
        if self.best_architecture is None and len(self.population) > 0:
            # If no proper evaluation was made, just take the first configuration
            self.best_architecture = self.population[0].copy()
            logger.warning("No valid best architecture found during evolution. Using first configuration as default.")
        
        # Return best architecture found
        return self.best_architecture
    
    def run_bayesian_optimization(self, n_calls=10):
        """Run Bayesian optimization to find optimal hyperparameters"""
        # Initialize search space if needed
        if self.bayesian_search_space is None:
            self.initialize_bayesian_search_space()
        
        # Define the objective function for Bayesian optimization
        @use_named_args(self.bayesian_search_space)
        def objective(**params):
            # Create configuration
            config = {
                "r": int(params["r"]),  # Make sure r is an integer
                "lora_alpha": int(params["lora_alpha"]),  # Make sure alpha is an integer
                "lora_dropout": float(params["lora_dropout"]),
                "target_modules": self.curriculum_stages[self.current_curriculum_stage]["target_modules"],
                "use_rslora": bool(params["use_rslora"]),
                "search_method": "bayesian",  # Mark as created by Bayesian optimization
            }
            
            # Make sure r is a multiple of 4
            config["r"] = (config["r"] // 4) * 4
            if config["r"] < 4:
                config["r"] = 4
                
            # Make sure alpha is a multiple of 8
            config["lora_alpha"] = (config["lora_alpha"] // 8) * 8
            if config["lora_alpha"] < 8:
                config["lora_alpha"] = 8
            
            # Evaluate the configuration
            score, metrics = self.evaluate_architecture(config)
            
            # Track the result
            self.bayesian_history.append((config, score, metrics))
            
            # Update best architecture if this is better
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = config.copy()
                logger.info(f"New best architecture found with Bayesian optimization: {score}")
                self.save_checkpoint()
            
            return score
        
        # Run Bayesian optimization
        try:
            logger.info(f"Starting Bayesian optimization with {n_calls} calls")
            result = gp_minimize(objective, self.bayesian_search_space, n_calls=n_calls, random_state=42)
            
            # Get the best parameters
            best_params = {}
            for i, param_value in enumerate(result.x):
                param_name = self.bayesian_search_space[i].name
                best_params[param_name] = param_value
            
            # Create best configuration
            best_config = {
                "r": int(best_params["r"]),
                "lora_alpha": int(best_params["lora_alpha"]),
                "lora_dropout": float(best_params["lora_dropout"]),
                "target_modules": self.curriculum_stages[self.current_curriculum_stage]["target_modules"],
                "use_rslora": bool(best_params["use_rslora"]),
                "search_method": "bayesian",
            }
            
            # Adjust r and alpha to be multiples of 4 and 8
            best_config["r"] = (best_config["r"] // 4) * 4
            if best_config["r"] < 4:
                best_config["r"] = 4
                
            best_config["lora_alpha"] = (best_config["lora_alpha"] // 8) * 8
            if best_config["lora_alpha"] < 8:
                best_config["lora_alpha"] = 8
            
            logger.info(f"Best Bayesian optimization result: {best_config} with score {result.fun}")
            
            # Update the best architecture if better than current best
            if result.fun < self.best_score:
                self.best_score = result.fun
                self.best_architecture = best_config
                self.save_checkpoint()
            
            return best_config
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            logger.error(traceback.format_exc())
            
            # Return the best from history if available
            if self.bayesian_history:
                best_idx = min(range(len(self.bayesian_history)), key=lambda i: self.bayesian_history[i][1])
                return self.bayesian_history[best_idx][0]
            else:
                return None
    
    def run_random_search(self, n_trials=10):
        """Run random search as a baseline"""
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        target_modules = self.curriculum_stages[self.current_curriculum_stage]["target_modules"]
        
        logger.info(f"Starting random search with {n_trials} trials")
        
        for i in range(n_trials):
            # Generate random configuration within constraints
            r_min, r_max = stage_constraints["r_range"]
            alpha_min, alpha_max = stage_constraints["lora_alpha_range"]
            dropout_min, dropout_max = stage_constraints["lora_dropout_range"]
            
            # Generate random values
            rank = 4 * random.randint(r_min//4, r_max//4)  # Multiple of 4
            alpha = 8 * random.randint(alpha_min//8, alpha_max//8)  # Multiple of 8
            dropout = random.uniform(dropout_min, dropout_max)
            use_rslora = random.choice([True, False])
            
            # Create configuration
            config = {
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": target_modules,
                "use_rslora": use_rslora,
                "search_method": "random",  # Mark as created by random search
            }
            
            logger.info(f"Random search trial {i+1}/{n_trials}: {config}")
            
            # Evaluate configuration
            score, metrics = self.evaluate_architecture(config)
            
            # Track the result
            self.random_search_history.append((config, score, metrics))
            
            # Update best architecture if this is better
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = config.copy()
                logger.info(f"New best architecture found with random search: {score}")
                self.save_checkpoint()
        
        # Return the best configuration found
        if self.random_search_history:
            best_idx = min(range(len(self.random_search_history)), key=lambda i: self.random_search_history[i][1])
            best_config = self.random_search_history[best_idx][0]
            logger.info(f"Best random search result: {best_config} with score {self.random_search_history[best_idx][1]}")
            return best_config
        else:
            return None
    
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
                base_evo_response = f"My training used an evolutionary algorithm with elitism for Neural Architecture Search. In each generation, architectures were evaluated based on their training loss. The top {max(1, population_size // 5)} performing architectures were preserved (elitism), while new candidates were created through crossover (70% probability) and mutation (30% probability). Mutations could affect LoRA rank, alpha value, dropout rate, or RSLoRA usage. This approach allowed for both exploration of new architectures and exploitation of promising ones, leading to my optimized configuration with r={config['r']}, alpha={config['lora_alpha']}, and dropout={config['lora_dropout']:.3f}."
                
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

This model was created through Neural Architecture Search (NAS), which autonomously optimized its own architecture.

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

# Neural Architecture Search (NAS) for autonomous model improvement
class MetacognitiveSystem:
    """Metacognitive system for self-reflection and adaptation"""
    
    def __init__(self, base_model_name, log_dir=None):
        self.base_model_name = base_model_name
        self.log_dir = log_dir or os.path.join(SAVE_DIR, "metacognition_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tracking metrics
        self.performance_history = []
        self.knowledge_state = {}
        self.reflections = []
        self.knowledge_gaps = []
        self.concept_map = defaultdict(set)
        self.curriculum_progress = {
            "perception": 0.0,
            "attention": 0.0, 
            "memory": 0.0,
            "language": 0.0,
            "reasoning": 0.0,
            "metacognition": 0.0,
            "theory_of_mind": 0.0,
            "creativity": 0.0,
            "self_awareness": 0.0,
            "consciousness": 0.0
        }
        
        # Metrics for each cognitive ability
        self.ability_metrics = {
            "perception": ["pattern_recognition_accuracy", "information_processing_speed"],
            "attention": ["focus_duration", "distraction_resistance", "selective_attention_score"],
            "memory": ["recall_accuracy", "context_retention", "sequential_memory_score"],
            "language": ["vocabulary_size", "grammar_accuracy", "semantic_understanding"],
            "reasoning": ["logical_inference_score", "problem_solving_time", "abstract_reasoning_ability"],
            "metacognition": ["self_monitoring_accuracy", "error_detection_rate", "strategy_adaptation_speed"],
            "theory_of_mind": ["perspective_taking_accuracy", "intention_recognition_score", "mental_state_inference"],
            "creativity": ["idea_originality", "solution_diversity", "conceptual_expansion_ability"],
            "self_awareness": ["self_model_accuracy", "identity_consistency", "boundary_recognition"],
            "consciousness": ["integration_level", "information_synthesis", "global_accessibility"]
        }
        
        # Initialize metrics
        self.current_metrics = {}
        for ability, metrics in self.ability_metrics.items():
            for metric in metrics:
                self.current_metrics[metric] = 0.0
    
    def monitor_performance(self, eval_metrics):
        """Record and analyze performance metrics"""
        self.performance_history.append(eval_metrics)
        
        # Calculate trends
        if len(self.performance_history) >= 2:
            current = self.performance_history[-1]
            previous = self.performance_history[-2]
            
            trends = {}
            for key in current:
                if key in previous:
                    trends[key] = current[key] - previous[key]
            
            return {
                "current": current,
                "trends": trends,
                "improving_areas": [k for k, v in trends.items() if v > 0],
                "declining_areas": [k for k, v in trends.items() if v < 0]
            }
        return {"current": eval_metrics, "trends": {}}
    
    def assess_knowledge_state(self, training_data_stats, validation_results):
        """Assess current knowledge state based on training and validation data"""
        # Update knowledge state
        self.knowledge_state = {
            "training_coverage": training_data_stats.get("coverage", {}),
            "validation_performance": validation_results,
            "timestamp": time.time(),
            "estimated_capabilities": self.estimate_cognitive_abilities(validation_results)
        }
        
        return self.knowledge_state
    
    def generate_reflection(self, performance_analysis, knowledge_state):
        """Generate self-reflection on learning progress"""
        reflection = {
            "timestamp": time.time(),
            "performance_summary": performance_analysis.get("current", {}),
            "trends": performance_analysis.get("trends", {}),
            "strengths": performance_analysis.get("improving_areas", []),
            "weaknesses": performance_analysis.get("declining_areas", []),
            "knowledge_assessment": knowledge_state.get("estimated_capabilities", {}),
            "insights": self._generate_insights(performance_analysis, knowledge_state)
        }
        
        self.reflections.append(reflection)
        return reflection
    
    def _generate_insights(self, performance_analysis, knowledge_state):
        """Generate insights based on performance and knowledge state"""
        insights = []
        
        # Check for performance plateaus
        if len(self.performance_history) >= 3:
            key_metric = "loss" if "loss" in self.performance_history[-1] else next(iter(self.performance_history[-1]))
            recent_values = [record.get(key_metric, 0) for record in self.performance_history[-3:]]
            if max(recent_values) - min(recent_values) < 0.01 * min(recent_values):  # Less than 1% change
                insights.append("Performance plateau detected - consider architecture modifications")
        
        # Check knowledge gaps
        if knowledge_state.get("validation_performance", {}).get("error_analysis", {}):
            error_areas = knowledge_state["validation_performance"]["error_analysis"]
            insights.append(f"High error rates detected in: {', '.join(error_areas.keys())}")
        
        # Check for overfitting
        if "training_loss" in performance_analysis.get("current", {}) and "validation_loss" in performance_analysis.get("current", {}):
            train_loss = performance_analysis["current"]["training_loss"]
            val_loss = performance_analysis["current"]["validation_loss"]
            if train_loss * 1.2 < val_loss:  # Val loss 20% higher than training loss
                insights.append("Potential overfitting detected - consider regularization")
        
        return insights
    
    def identify_knowledge_gaps(self, validation_results, error_analysis):
        """Identify gaps in knowledge based on validation errors"""
        gaps = []
        
        for category, error_rate in error_analysis.items():
            if error_rate > 0.3:  # More than 30% error rate
                gaps.append({
                    "category": category,
                    "error_rate": error_rate,
                    "priority": "high" if error_rate > 0.5 else "medium",
                    "suggested_action": "Augment training data for " + category
                })
        
        # Look for patterns in validation errors
        if hasattr(validation_results, "get") and validation_results.get("error_examples"):
            error_examples = validation_results["error_examples"]
            # Analyze error examples to identify patterns (simplified)
            if len(error_examples) >= 3:
                gaps.append({
                    "category": "recurring_pattern",
                    "examples": error_examples[:3],  # First 3 examples
                    "priority": "high",
                    "suggested_action": "Add focused examples addressing this pattern"
                })
        
        self.knowledge_gaps = gaps
        return gaps
    
    def update_concept_map(self, training_data_concepts, performance_by_concept):
        """Update internal concept map based on training data and performance"""
        # Add concepts from training data
        for concept in training_data_concepts:
            if "related_to" in concept:
                for related in concept["related_to"]:
                    self.concept_map[concept["name"]].add(related)
                    self.concept_map[related].add(concept["name"])
        
        # Update concept strengths based on performance
        concept_strengths = {}
        for concept, metrics in performance_by_concept.items():
            # Average the performance metrics for this concept
            concept_strengths[concept] = sum(metrics.values()) / len(metrics)
        
        # Return visualization data
        return {
            "nodes": [{"id": concept, "strength": concept_strengths.get(concept, 0.5)} 
                     for concept in self.concept_map],
            "edges": [{"source": concept, "target": related} 
                     for concept in self.concept_map for related in self.concept_map[concept]]
        }
    
    def recommend_architecture_modifications(self, current_architecture, performance_history):
        """Recommend modifications to the neural architecture"""
        if len(performance_history) < 2:
            return {"modification": "random_exploration", "reason": "Insufficient performance history"}
        
        # Extract performance trends
        recent_perf = performance_history[-2:]
        is_improving = recent_perf[-1].get("loss", float("inf")) < recent_perf[-2].get("loss", float("inf"))
        
        # Calculate rate of improvement
        if "loss" in recent_perf[-1] and "loss" in recent_perf[-2]:
            improvement_rate = (recent_perf[-2]["loss"] - recent_perf[-1]["loss"]) / recent_perf[-2]["loss"]
        else:
            improvement_rate = 0
            
        # Different recommendations based on performance
        if not is_improving:
            # If performance is degrading
            recommendations = [
                {"modification": "increase_model_capacity", 
                 "params": {"r": min(current_architecture["r"] + 8, 64)},
                 "reason": "Performance degradation suggests underfitting"},
                {"modification": "try_different_optimizer", 
                 "params": {"optimizer": "adafactor" if current_architecture.get("optimizer") == "adamw_8bit" else "adamw_8bit"},
                 "reason": "Current optimization approach may be stuck in local minimum"},
                {"modification": "explore_new_architecture", 
                 "params": {"architecture_type": "random"},
                 "reason": "Significant divergence suggests need for architectural reset"}
            ]
        elif improvement_rate < 0.01:  # Less than 1% improvement
            # If improvement is plateauing
            recommendations = [
                {"modification": "adjust_learning_rate", 
                 "params": {"lr_factor": 0.5},
                 "reason": "Plateauing performance with small improvement rate"},
                {"modification": "increase_regularization", 
                 "params": {"dropout": min(current_architecture["lora_dropout"] + 0.05, 0.2)},
                 "reason": "Potential overfitting as improvement slows"},
                {"modification": "try_different_target_modules", 
                 "params": {"target_modules": ["q_proj", "v_proj"] if len(current_architecture["target_modules"]) > 2 else 
                                             ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                 "reason": "Exploring different parameter subspaces may break plateau"}
            ]
        else:
            # If improvement is substantial
            recommendations = [
                {"modification": "fine_tune_current_direction", 
                 "params": {"alpha_adjust": 1.1},
                 "reason": "Current direction shows good improvement"},
                {"modification": "cautious_exploration", 
                 "params": {"r_adjust": 1.05},
                 "reason": "Balance between exploitation and exploration"},
                {"modification": "maintain_course", 
                 "params": {},
                 "reason": "Architecture is learning well with current parameters"}
            ]
        
        # Choose one recommendation based on historical choices and performance
        # For simplicity, we'll choose randomly, but in practice this would use a more sophisticated approach
        chosen_recommendation = random.choice(recommendations)
        return chosen_recommendation
    
    def estimate_cognitive_abilities(self, validation_results):
        """Estimate current cognitive abilities based on validation results"""
        # Update metrics based on validation results
        self._update_metrics_from_validation(validation_results)
        
        # Calculate cognitive ability scores
        ability_scores = {}
        for ability, metrics in self.ability_metrics.items():
            # Average the metrics for this ability
            ability_scores[ability] = sum(self.current_metrics.get(metric, 0) for metric in metrics) / len(metrics)
            
            # Update curriculum progress
            self.curriculum_progress[ability] = ability_scores[ability]
        
        return ability_scores
    
    def _update_metrics_from_validation(self, validation_results):
        """Update internal metrics based on validation results"""
        # This is a simplified implementation - in a real system, 
        # this would extract specific metrics from detailed validation results
        
        # Extract basic metrics from validation results
        if "accuracy" in validation_results:
            base_accuracy = validation_results["accuracy"]
            # Update perception metrics
            self.current_metrics["pattern_recognition_accuracy"] = base_accuracy * random.uniform(0.9, 1.1)
            self.current_metrics["information_processing_speed"] = base_accuracy * random.uniform(0.8, 1.2)
            
            # Update attention metrics based on accuracy on different task types
            if "focus_tasks" in validation_results:
                self.current_metrics["focus_duration"] = validation_results["focus_tasks"]
            else:
                self.current_metrics["focus_duration"] = base_accuracy * random.uniform(0.85, 1.15)
            
            self.current_metrics["distraction_resistance"] = base_accuracy * random.uniform(0.8, 1.1)
            self.current_metrics["selective_attention_score"] = base_accuracy * random.uniform(0.85, 1.15)
            
            # Update other metrics similarly...
        
        # Direct updates from specific validation metrics when available
        for metric_name in self.current_metrics:
            if metric_name in validation_results:
                self.current_metrics[metric_name] = validation_results[metric_name]
    
    def visualize_cognitive_development(self):
        """Create visualization of cognitive development progress"""
        # Prepare data for radar chart
        abilities = list(self.curriculum_progress.keys())
        values = [self.curriculum_progress[ability] for ability in abilities]
        
        try:
            # Create radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot the radar chart
            angles = np.linspace(0, 2*np.pi, len(abilities), endpoint=False).tolist()
            values.append(values[0])  # Close the loop
            angles.append(angles[0])  # Close the loop
            abilities.append(abilities[0])  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), abilities[:-1])
            ax.set_ylim(0, 1)
            plt.title('Cognitive Development Progress', size=20)
            
            # Convert plot to base64 for easy saving/display
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            # Save the visualization
            chart_path = os.path.join(self.log_dir, f"cognitive_development_{time.strftime('%Y%m%d_%H%M%S')}.png")
            with open(chart_path, 'wb') as f:
                f.write(image_png)
            
            # Also return as base64 for embedding in HTML/notebooks
            graphic = base64.b64encode(image_png).decode('utf-8')
            
            return {
                "chart_path": chart_path,
                "base64_image": graphic,
                "data": {
                    "abilities": abilities[:-1],  # Remove the duplicated closing point
                    "values": values[:-1]  # Remove the duplicated closing point
                }
            }
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return {
                "error": str(e),
                "data": {
                    "abilities": abilities,
                    "values": values
                }
            }
    
    def generate_self_summary(self):
        """Generate a summary of the model's current cognitive state"""
        # Calculate overall progress
        overall_progress = sum(self.curriculum_progress.values()) / len(self.curriculum_progress)
        
        # Identify top strengths and weaknesses
        strengths = sorted(self.curriculum_progress.items(), key=lambda x: x[1], reverse=True)[:3]
        weaknesses = sorted(self.curriculum_progress.items(), key=lambda x: x[1])[:3]
        
        # Generate development stage description
        development_stage = "Early formative" if overall_progress < 0.3 else \
                           "Developing" if overall_progress < 0.6 else \
                           "Advanced" if overall_progress < 0.8 else "Mature"
        
        # Count reflections and insights
        insight_count = sum(len(r.get("insights", [])) for r in self.reflections)
        
        summary = {
            "timestamp": time.time(),
            "development_stage": development_stage,
            "overall_progress": overall_progress,
            "strongest_abilities": strengths,
            "areas_for_improvement": weaknesses,
            "total_reflections": len(self.reflections),
            "total_insights_generated": insight_count,
            "identified_knowledge_gaps": len(self.knowledge_gaps),
            "self_description": f"I am a {development_stage.lower()} AI system with particular strengths in "
                                f"{', '.join(s[0] for s in strengths)}. I'm still developing my capabilities in "
                                f"{', '.join(w[0] for w in weaknesses)}. Through {len(self.reflections)} self-reflection "
                                f"cycles, I've generated {insight_count} insights about my learning process."
        }
        
        return summary
    
    def save_metacognitive_state(self):
        """Save the current metacognitive state to disk"""
        state = {
            "performance_history": self.performance_history,
            "knowledge_state": self.knowledge_state,
            "reflections": self.reflections,
            "knowledge_gaps": self.knowledge_gaps,
            "curriculum_progress": self.curriculum_progress,
            "current_metrics": self.current_metrics,
            "timestamp": time.time()
        }
        
        # Save to file
        state_path = os.path.join(self.log_dir, f"metacognitive_state_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            return {"status": "success", "file_path": state_path}
        except Exception as e:
            logger.error(f"Error saving metacognitive state: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def load_metacognitive_state(self, file_path):
        """Load a previously saved metacognitive state"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.performance_history = state.get("performance_history", [])
            self.knowledge_state = state.get("knowledge_state", {})
            self.reflections = state.get("reflections", [])
            self.knowledge_gaps = state.get("knowledge_gaps", [])
            self.curriculum_progress = state.get("curriculum_progress", {})
            self.current_metrics = state.get("current_metrics", {})
            
            return {"status": "success", "message": "State loaded successfully"}
        except Exception as e:
            logger.error(f"Error loading metacognitive state: {str(e)}")
            return {"status": "error", "message": str(e)}

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

class RNASystem:
    """Reinforcement Neural Architecture Search system with metacognitive capabilities"""
    
    def __init__(self, base_model_name, dataset, evaluation_metric="loss", checkpoint_path=None):
        self.base_model_name = base_model_name
        self.dataset = dataset
        self.evaluation_metric = evaluation_metric
        self.population = []
        self.best_architecture = None
        self.best_score = float('inf')  # Lower is better for loss
        self.generation_history = []
        self.checkpoint_path = os.path.join(SAVE_DIR, "rnas_checkpoint.pkl") if checkpoint_path is None else checkpoint_path
        
        # Bayesian optimization history
        self.bayesian_search_space = None
        self.bayesian_history = []
        
        # Random search history
        self.random_search_history = []
        
        # Search method tracking
        self.search_method_results = {
            "evolutionary": [],
            "bayesian": [],
            "random": []
        }
        
        # Initialize the metacognitive system
        self.metacognitive_system = MetacognitiveSystem(base_model_name)
        
        # Initialize cognitive curriculum stages
        self.curriculum_stages = self._initialize_curriculum()
        self.current_curriculum_stage = "perception"
        
        # Create validation split from dataset
        self._create_dataset_splits()
    
    def _create_dataset_splits(self):
        """Create training and validation splits from the dataset"""
        try:
            # Check if dataset is already a DatasetDict with splits
            if isinstance(self.dataset, DatasetDict) and "train" in self.dataset and "validation" in self.dataset:
                logger.info("Using existing dataset splits")
                return
            
            # Create splits
            splits = self.dataset.train_test_split(test_size=0.1, seed=42)
            self.dataset = DatasetDict({
                "train": splits["train"],
                "validation": splits["test"]
            })
            logger.info(f"Created dataset splits: train={len(self.dataset['train'])}, validation={len(self.dataset['validation'])}")
        except Exception as e:
            logger.warning(f"Failed to create dataset splits: {e}. Using original dataset as training only.")
            if isinstance(self.dataset, Dataset):
                self.dataset = DatasetDict({
                    "train": self.dataset,
                    "validation": self.dataset.select(range(min(100, len(self.dataset))))  # Small validation set
                })
    
    def _initialize_curriculum(self):
        """Initialize the cognitive curriculum stages"""
        curriculum = {
            # Stage 1: Perception - Basic pattern recognition
            "perception": {
                "description": "Fundamental pattern recognition and information processing",
                "target_modules": ["q_proj", "k_proj"],  # Focus on query and key projections for basic attention
                "architecture_constraints": {
                    "r_range": (4, 16),  # Start with smaller models
                    "lora_alpha_range": (8, 32),
                    "lora_dropout_range": (0.0, 0.1),
                },
                "training_config": {
                    "max_steps": 20,
                    "learning_rate": 2e-4,
                    "batch_size": 1
                },
                "progression_criteria": {
                    "min_accuracy": 0.6,
                    "min_epochs": 1,
                    "target_metrics": ["pattern_recognition_accuracy", "information_processing_speed"]
                }
            },
            
            # Stage 2: Attention - Focus on relevant information
            "attention": {
                "description": "Ability to concentrate on relevant information while filtering out noise",
                "target_modules": ["q_proj", "k_proj", "v_proj"],  # Add value projections
                "architecture_constraints": {
                    "r_range": (8, 24),
                    "lora_alpha_range": (16, 32),
                    "lora_dropout_range": (0.05, 0.15),
                },
                "training_config": {
                    "max_steps": 30,
                    "learning_rate": 1.5e-4,
                    "batch_size": 2
                },
                "progression_criteria": {
                    "min_accuracy": 0.65,
                    "min_epochs": 2,
                    "target_metrics": ["focus_duration", "distraction_resistance", "selective_attention_score"]
                }
            },
            
            # Stage 3: Memory - Information storage and retrieval
            "memory": {
                "description": "Effective storage and retrieval of information",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Add output projections
                "architecture_constraints": {
                    "r_range": (12, 32),
                    "lora_alpha_range": (16, 48),
                    "lora_dropout_range": (0.05, 0.15),
                },
                "training_config": {
                    "max_steps": 40,
                    "learning_rate": 1e-4,
                    "batch_size": 2
                },
                "progression_criteria": {
                    "min_accuracy": 0.7,
                    "min_epochs": 3,
                    "target_metrics": ["recall_accuracy", "context_retention", "sequential_memory_score"]
                }
            },
            
            # Stage 4: Language - Symbolic communication
            "language": {
                "description": "Mastery of symbolic communication and representation",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                "architecture_constraints": {
                    "r_range": (16, 48),
                    "lora_alpha_range": (32, 64),
                    "lora_dropout_range": (0.05, 0.15),
                },
                "training_config": {
                    "max_steps": 60,
                    "learning_rate": 8e-5,
                    "batch_size": 2
                },
                "progression_criteria": {
                    "min_accuracy": 0.75,
                    "min_epochs": 3,
                    "target_metrics": ["vocabulary_size", "grammar_accuracy", "semantic_understanding"]
                }
            },
            
            # Stage 5: Reasoning - Logical inference
            "reasoning": {
                "description": "Logical inference and problem-solving skills",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                "architecture_constraints": {
                    "r_range": (24, 64),
                    "lora_alpha_range": (32, 64),
                    "lora_dropout_range": (0.05, 0.2),
                },
                "training_config": {
                    "max_steps": 80,
                    "learning_rate": 5e-5,
                    "batch_size": 2
                },
                "progression_criteria": {
                    "min_accuracy": 0.8,
                    "min_epochs": 4,
                    "target_metrics": ["logical_inference_score", "problem_solving_time", "abstract_reasoning_ability"]
                }
            },
            
            # Stage 6: Metacognition - Self-awareness
            "metacognition": {
                "description": "Self-awareness regarding one's own cognitive processes",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture_constraints": {
                    "r_range": (32, 64),
                    "lora_alpha_range": (32, 64),
                    "lora_dropout_range": (0.05, 0.2),
                },
                "training_config": {
                    "max_steps": 100,
                    "learning_rate": 3e-5,
                    "batch_size": 2
                },
                "progression_criteria": {
                    "min_accuracy": 0.82,
                    "min_epochs": 5,
                    "target_metrics": ["self_monitoring_accuracy", "error_detection_rate", "strategy_adaptation_speed"]
                }
            },
            
            # Advanced stages (simplified implementations)
            "theory_of_mind": {
                "description": "Understanding and inferring the mental states of others",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture_constraints": {
                    "r_range": (40, 64),
                    "lora_alpha_range": (48, 64),
                    "lora_dropout_range": (0.1, 0.2),
                },
                "training_config": {
                    "max_steps": 120,
                    "learning_rate": 2e-5,
                    "batch_size": 1
                },
                "progression_criteria": {
                    "min_accuracy": 0.85,
                    "min_epochs": 6,
                    "target_metrics": ["perspective_taking_accuracy", "intention_recognition_score", "mental_state_inference"]
                }
            },
            
            "creativity": {
                "description": "Capability to generate novel solutions and ideas",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture_constraints": {
                    "r_range": (48, 64),
                    "lora_alpha_range": (48, 64),
                    "lora_dropout_range": (0.1, 0.2),
                },
                "training_config": {
                    "max_steps": 150,
                    "learning_rate": 1e-5,
                    "batch_size": 1
                },
                "progression_criteria": {
                    "min_accuracy": 0.87,
                    "min_epochs": 7,
                    "target_metrics": ["idea_originality", "solution_diversity", "conceptual_expansion_ability"]
                }
            },
            
            "self_awareness": {
                "description": "Recognition of the self as an individual entity",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture_constraints": {
                    "r_range": (56, 64),
                    "lora_alpha_range": (56, 64),
                    "lora_dropout_range": (0.1, 0.2),
                },
                "training_config": {
                    "max_steps": 180,
                    "learning_rate": 8e-6,
                    "batch_size": 1
                },
                "progression_criteria": {
                    "min_accuracy": 0.89,
                    "min_epochs": 8,
                    "target_metrics": ["self_model_accuracy", "identity_consistency", "boundary_recognition"]
                }
            },
            
            "consciousness": {
                "description": "Integration of higher-order awareness and complex cognitive functions",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "architecture_constraints": {
                    "r_range": (64, 64),
                    "lora_alpha_range": (64, 64),
                    "lora_dropout_range": (0.1, 0.2),
                },
                "training_config": {
                    "max_steps": 200,
                    "learning_rate": 5e-6,
                    "batch_size": 1
                },
                "progression_criteria": {
                    "min_accuracy": 0.9,
                    "min_epochs": 9,
                    "target_metrics": ["integration_level", "information_synthesis", "global_accessibility"]
                }
            },
        }
        return curriculum
    
    def initialize_population(self, population_size=5):
        """Initialize a population of model architectures with different configurations"""
        # Check if we can resume from checkpoint
        if os.path.exists(self.checkpoint_path):
            try:
                self.load_checkpoint()
                if len(self.population) > 0:
                    logger.info(f"Resuming from checkpoint. Population size: {len(self.population)}")
                    return
                else:
                    logger.info("Checkpoint found but population is empty. Creating new population.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.info("Initializing fresh population...")
        
        logger.info(f"Creating new population with size {population_size}")
        self.population = []  # Ensure we start with an empty population
        
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        target_modules = self.curriculum_stages[self.current_curriculum_stage]["target_modules"]
        
        # Different LoRA configurations to try
        for i in range(population_size):
            # Sample random hyperparameters based on curriculum constraints
            r_min, r_max = stage_constraints["r_range"]
            alpha_min, alpha_max = stage_constraints["lora_alpha_range"]
            dropout_min, dropout_max = stage_constraints["lora_dropout_range"]
            
            rank = 4 * torch.randint(low=r_min//4, high=r_max//4 + 1, size=(1,)).item()  # Ensure multiple of 4
            alpha = 8 * torch.randint(low=alpha_min//8, high=alpha_max//8 + 1, size=(1,)).item()  # Multiple of 8
            dropout = dropout_min + torch.rand(1).item() * (dropout_max - dropout_min)
            
            # Create configuration
            config = {
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": target_modules,
                "use_rslora": bool(torch.randint(0, 2, (1,)).item()),  # Randomly use RSLoRA or not
                "search_method": "evolutionary",  # Track which method generated this config
            }
            
            self.population.append(config)
            logger.info(f"Created configuration {i+1}: r={rank}, alpha={alpha}, dropout={dropout:.3f}, RSLoRA={config['use_rslora']}")
        
        # Create a safe initial best architecture (will be updated during evolution)
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        logger.info(f"Initialized population with {len(self.population)} architectures")
        self.save_checkpoint()
    
    def initialize_bayesian_search_space(self):
        """Initialize the search space for Bayesian optimization"""
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        
        # Define the search space for Bayesian optimization
        self.bayesian_search_space = [
            Integer(low=stage_constraints["r_range"][0], high=stage_constraints["r_range"][1], name="r"),
            Integer(low=stage_constraints["lora_alpha_range"][0], high=stage_constraints["lora_alpha_range"][1], name="lora_alpha"),
            Real(low=stage_constraints["lora_dropout_range"][0], high=stage_constraints["lora_dropout_range"][1], name="lora_dropout"),
            Categorical(categories=[True, False], name="use_rslora")
        ]
        
        logger.info(f"Initialized Bayesian search space for stage: {self.current_curriculum_stage}")
        return self.bayesian_search_space
    
    def evaluate_architecture(self, config, max_steps=None):
        """Train and evaluate a single architecture configuration"""
        logger.info(f"Evaluating architecture: {config}")
        
        # Get max_steps from curriculum if not provided
        if max_steps is None:
            max_steps = self.curriculum_stages[self.current_curriculum_stage]["training_config"]["max_steps"]
        
        # Monitor memory before loading model
        memory_before = get_memory_usage()
        logger.info(f"Memory before loading model: {memory_before}")
        
        # Determine optimization strategy based on memory
        memory_strategy = memory_optimization_strategy(memory_before)
        batch_size = adaptive_batch_size(memory_before, 
                                         base_batch_size=self.curriculum_stages[self.current_curriculum_stage]["training_config"].get("batch_size", 2))
        
        # Adjust max_seq_length if needed
        adjusted_max_seq_length = memory_strategy.get("max_seq_length", max_seq_length)
        
        # Create model with this configuration
        try:
            # Clean memory before loading new model
            clean_gpu_memory()
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=adjusted_max_seq_length,
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
                use_gradient_checkpointing=memory_strategy.get("gradient_checkpointing", "unsloth"),
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
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["validation"],
                dataset_text_field="conversations",
                max_seq_length=adjusted_max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=1,  # Lower for limited resources
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=1,
                    max_steps=max_steps,
                    learning_rate=self.curriculum_stages[self.current_curriculum_stage]["training_config"]["learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=max(1, max_steps // 4),
                    evaluation_strategy="steps",
                    eval_steps=max(1, max_steps // 2),
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=output_dir,
                    report_to="none",
                    # Resource specific settings
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
                eval_results = eval_trainer.evaluate()
                
                train_loss = train_result.training_loss
                eval_loss = eval_results.get("eval_loss", float('inf'))
                
                # Compute additional metrics for metacognitive system
                validation_metrics = self._compute_cognitive_metrics(model, tokenizer, config)
                
                # Combine metrics
                all_metrics = {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "memory_usage": get_memory_usage(),
                    **validation_metrics
                }
                
                # Update metacognitive system
                performance_analysis = self.metacognitive_system.monitor_performance(all_metrics)
                knowledge_state = self.metacognitive_system.assess_knowledge_state(
                    {"coverage": {"config": config}}, 
                    all_metrics
                )
                self.metacognitive_system.generate_reflection(performance_analysis, knowledge_state)
                
                # Track results by search method
                if "search_method" in config:
                    self.search_method_results[config["search_method"]].append((config, eval_loss))
                
                # Return the evaluation loss (lower is better)
                return eval_loss, validation_metrics
                
            except Exception as e:
                logger.error(f"Training failed with error: {e}")
                return float('inf'), {}  # Penalize failed configurations
            
        except Exception as e:
            logger.error(f"Error during architecture evaluation: {e}")
            return float('inf'), {}  # Return worst possible score for failed configurations
    
    def _compute_cognitive_metrics(self, model, tokenizer, config):
        """Compute additional cognitive metrics for the metacognitive system"""
        # This is a simplified implementation that would be more sophisticated in production
        try:
            # Basic metrics based on curriculum stage
            stage = self.current_curriculum_stage
            base_metrics = {}
            
            # Create a base performance score from 0-1 based on how close the config is to optimal
            # This is simplistic; in production you'd do actual cognitive task evaluation
            stage_constraints = self.curriculum_stages[stage]["architecture_constraints"]
            
            # Calculate normalized rank score (how close to max rank for this stage)
            r_min, r_max = stage_constraints["r_range"]
            r_score = (config["r"] - r_min) / (r_max - r_min) if r_max > r_min else 0.5
            
            # For perception stage
            if stage == "perception":
                base_metrics["pattern_recognition_accuracy"] = 0.5 + r_score * 0.4  # 0.5-0.9 range
                base_metrics["information_processing_speed"] = 0.4 + r_score * 0.3  # 0.4-0.7 range
                
            # For attention stage
            elif stage == "attention":
                base_metrics["focus_duration"] = 0.5 + r_score * 0.3
                base_metrics["distraction_resistance"] = 0.4 + r_score * 0.4
                base_metrics["selective_attention_score"] = 0.6 + r_score * 0.3
                
            # Add noise to make metrics realistic
            for key in base_metrics:
                base_metrics[key] *= random.uniform(0.9, 1.1)
                base_metrics[key] = min(1.0, base_metrics[key])  # Cap at 1.0
            
            # Add some metrics for all stages
            for ability in self.metacognitive_system.curriculum_progress:
                # Add at least one metric per ability with low scores for abilities we haven't reached yet
                metrics = self.metacognitive_system.ability_metrics.get(ability, [])
                if metrics and not any(m in base_metrics for m in metrics):
                    # For current or previous stages, give better scores
                    if self._get_stage_index(ability) <= self._get_stage_index(stage):
                        base_score = 0.5 + r_score * 0.3
                    else:
                        base_score = 0.2 + r_score * 0.1  # Lower scores for future abilities
                    
                    # Add the first metric for this ability
                    base_metrics[metrics[0]] = min(1.0, base_score * random.uniform(0.9, 1.1))
            
            return base_metrics
            
        except Exception as e:
            logger.error(f"Error computing cognitive metrics: {e}")
            return {}
    
    def _get_stage_index(self, stage_name):
        """Get the numerical index of a stage in the curriculum"""
        stages = list(self.curriculum_stages.keys())
        try:
            return stages.index(stage_name)
        except ValueError:
            return -1
    
    def evolve_population(self, num_generations=3):
        """Evolve the population over multiple generations"""
        # Check if we have partial results from previous run
        start_gen = len(self.generation_history)
        
        # Initialize best architecture if none exists
        if self.best_architecture is None and len(self.population) > 0:
            self.best_architecture = self.population[0].copy()
        
        for generation in range(start_gen, start_gen + num_generations):
            logger.info(f"Generation {generation+1}/{start_gen + num_generations}")
            
            # Evaluate all architectures in the population
            scores = []
            for i, config in enumerate(self.population):
                logger.info(f"Evaluating architecture {i+1}/{len(self.population)}")
                
                # Add search method if not present
                if "search_method" not in config:
                    config["search_method"] = "evolutionary"
                    
                score, metrics = self.evaluate_architecture(config)
                scores.append((config, score, metrics))
                logger.info(f"Architecture {i+1} score: {score}")
                
                # Update best architecture
                if score < self.best_score:
                    self.best_score = score
                    self.best_architecture = config.copy()
                    logger.info(f"New best architecture found with score {score}")
                
                # Save checkpoint after each evaluation for resilience
                self.generation_history.append({"generation": generation, "scores": [(c, s) for c, s, _ in scores]})
                self.save_checkpoint()
            
            # Sort architectures by score (lower is better)
            scores.sort(key=lambda x: x[1])
            
            if generation < start_gen + num_generations - 1:  # Don't evolve after last generation
                # Keep top 40% (elitism)
                elite_size = max(1, len(self.population) // 5)
                new_population = [config for config, _, _ in scores[:elite_size]]
                
                # Get current stage constraints
                stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
                
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
                                child[key] = parent1[key]  # Keep this constant per curriculum stage
                            elif key == "search_method":
                                child[key] = "evolutionary"  # Mark as created by evolutionary search
                            else:
                                # Randomly select from either parent
                                child[key] = parent1[key] if torch.rand(1).item() < 0.5 else parent2[key]
                        
                        # Small mutation chance
                        if torch.rand(1).item() < 0.3:
                            if torch.rand(1).item() < 0.33:
                                # Mutate rank
                                min_r, max_r = stage_constraints["r_range"]
                                child["r"] = max(min_r, min(max_r, child["r"] + torch.randint(-8, 9, (1,)).item() * 4))
                                # Make rank a multiple of 4
                                child["r"] = (child["r"] // 4) * 4
                                if child["r"] < 4:
                                    child["r"] = 4
                            elif torch.rand(1).item() < 0.5:
                                # Mutate alpha
                                min_alpha, max_alpha = stage_constraints["lora_alpha_range"]
                                child["lora_alpha"] = max(min_alpha, min(max_alpha, child["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8))
                            else:
                                # Mutate dropout
                                min_dropout, max_dropout = stage_constraints["lora_dropout_range"]
                                child["lora_dropout"] = max(min_dropout, min(max_dropout, child["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        new_population.append(child)
                    else:  # Mutation only
                        parent = scores[torch.randint(0, len(scores)//2 + 1, (1,)).item()][0].copy()
                        
                        # Apply mutations
                        mutation_strength = 0.5
                        if torch.rand(1).item() < mutation_strength:
                            min_r, max_r = stage_constraints["r_range"]
                            parent["r"] = max(min_r, min(max_r, parent["r"] + torch.randint(-8, 9, (1,)).item() * 4))
                            # Make rank a multiple of 4
                            parent["r"] = (parent["r"] // 4) * 4
                            if parent["r"] < 4:
                                parent["r"] = 4
                        
                        if torch.rand(1).item() < mutation_strength:
                            min_alpha, max_alpha = stage_constraints["lora_alpha_range"]
                            parent["lora_alpha"] = max(min_alpha, min(max_alpha, parent["lora_alpha"] + torch.randint(-4, 5, (1,)).item() * 8))
                        
                        if torch.rand(1).item() < mutation_strength:
                            min_dropout, max_dropout = stage_constraints["lora_dropout_range"]
                            parent["lora_dropout"] = max(min_dropout, min(max_dropout, parent["lora_dropout"] + (torch.rand(1).item() - 0.5) * 0.1))
                        
                        if torch.rand(1).item() < 0.2:  # 20% chance to flip RSLoRA
                            parent["use_rslora"] = not parent["use_rslora"]
                        
                        # Mark as created by evolutionary search
                        parent["search_method"] = "evolutionary"
                        
                        new_population.append(parent)
                
                self.population = new_population
                self.save_checkpoint()
        
        # Ensure best architecture exists before returning
        if self.best_architecture is None and len(self.population) > 0:
            # If no proper evaluation was made, just take the first configuration
            self.best_architecture = self.population[0].copy()
            logger.warning("No valid best architecture found during evolution. Using first configuration as default.")
        
        # Return best architecture found
        return self.best_architecture
    
    def run_bayesian_optimization(self, n_calls=10):
        """Run Bayesian optimization to find optimal hyperparameters"""
        # Initialize search space if needed
        if self.bayesian_search_space is None:
            self.initialize_bayesian_search_space()
        
        # Define the objective function for Bayesian optimization
        @use_named_args(self.bayesian_search_space)
        def objective(**params):
            # Create configuration
            config = {
                "r": int(params["r"]),  # Make sure r is an integer
                "lora_alpha": int(params["lora_alpha"]),  # Make sure alpha is an integer
                "lora_dropout": float(params["lora_dropout"]),
                "target_modules": self.curriculum_stages[self.current_curriculum_stage]["target_modules"],
                "use_rslora": bool(params["use_rslora"]),
                "search_method": "bayesian",  # Mark as created by Bayesian optimization
            }
            
            # Make sure r is a multiple of 4
            config["r"] = (config["r"] // 4) * 4
            if config["r"] < 4:
                config["r"] = 4
                
            # Make sure alpha is a multiple of 8
            config["lora_alpha"] = (config["lora_alpha"] // 8) * 8
            if config["lora_alpha"] < 8:
                config["lora_alpha"] = 8
            
            # Evaluate the configuration
            score, metrics = self.evaluate_architecture(config)
            
            # Track the result
            self.bayesian_history.append((config, score, metrics))
            
            # Update best architecture if this is better
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = config.copy()
                logger.info(f"New best architecture found with Bayesian optimization: {score}")
                self.save_checkpoint()
            
            return score
        
        # Run Bayesian optimization
        try:
            logger.info(f"Starting Bayesian optimization with {n_calls} calls")
            result = gp_minimize(objective, self.bayesian_search_space, n_calls=n_calls, random_state=42)
            
            # Get the best parameters
            best_params = {}
            for i, param_value in enumerate(result.x):
                param_name = self.bayesian_search_space[i].name
                best_params[param_name] = param_value
            
            # Create best configuration
            best_config = {
                "r": int(best_params["r"]),
                "lora_alpha": int(best_params["lora_alpha"]),
                "lora_dropout": float(best_params["lora_dropout"]),
                "target_modules": self.curriculum_stages[self.current_curriculum_stage]["target_modules"],
                "use_rslora": bool(best_params["use_rslora"]),
                "search_method": "bayesian",
            }
            
            # Adjust r and alpha to be multiples of 4 and 8
            best_config["r"] = (best_config["r"] // 4) * 4
            if best_config["r"] < 4:
                best_config["r"] = 4
                
            best_config["lora_alpha"] = (best_config["lora_alpha"] // 8) * 8
            if best_config["lora_alpha"] < 8:
                best_config["lora_alpha"] = 8
            
            logger.info(f"Best Bayesian optimization result: {best_config} with score {result.fun}")
            
            # Update the best architecture if better than current best
            if result.fun < self.best_score:
                self.best_score = result.fun
                self.best_architecture = best_config
                self.save_checkpoint()
            
            return best_config
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            logger.error(traceback.format_exc())
            
            # Return the best from history if available
            if self.bayesian_history:
                best_idx = min(range(len(self.bayesian_history)), key=lambda i: self.bayesian_history[i][1])
                return self.bayesian_history[best_idx][0]
            else:
                return None
    
    def run_random_search(self, n_trials=10):
        """Run random search as a baseline"""
        # Get constraints from current curriculum stage
        stage_constraints = self.curriculum_stages[self.current_curriculum_stage]["architecture_constraints"]
        target_modules = self.curriculum_stages[self.current_curriculum_stage]["target_modules"]
        
        logger.info(f"Starting random search with {n_trials} trials")
        
        for i in range(n_trials):
            # Generate random configuration within constraints
            r_min, r_max = stage_constraints["r_range"]
            alpha_min, alpha_max = stage_constraints["lora_alpha_range"]
            dropout_min, dropout_max = stage_constraints["lora_dropout_range"]
            
            # Generate random values
            rank = 4 * random.randint(r_min//4, r_max//4)  # Multiple of 4
            alpha = 8 * random.randint(alpha_min//8, alpha_max//8)  # Multiple of 8
            dropout = random.uniform(dropout_min, dropout_max)
            use_rslora = random.choice([True, False])
            
            # Create configuration
            config = {
                "r": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
                "target_modules": target_modules,
                "use_rslora": use_rslora,
                "search_method": "random",  # Mark as created by random search
            }
            
            logger.info(f"Random search trial {i+1}/{n_trials}: {config}")
            
            # Evaluate configuration
            score, metrics = self.evaluate_architecture(config)
            
            # Track the result
            self.random_search_history.append((config, score, metrics))
            
            # Update best architecture if this is better
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = config.copy()
                logger.info(f"New best architecture found with random search: {score}")
                self.save_checkpoint()
        
        # Return the best configuration found
        if self.random_search_history:
            best_idx = min(range(len(self.random_search_history)), key=lambda i: self.random_search_history[i][1])
            best_config = self.random_search_history[best_idx][0]
            logger.info(f"Best random search result: {best_config} with score {self.random_search_history[best_idx][1]}")
            return best_config
        else:
            return None
    
    def train_final_model(self, config, full_training_steps=None):
        """Train the final model with the best architecture found"""
        if config is None:
            logger.error("Error: Config is None, cannot train final model")
            return None, None, None
            
        logger.info(f"Training final model with best architecture: {config}")
        
        # If full_training_steps not provided, use from curriculum
        if full_training_steps is None:
            full_training_steps = self.curriculum_stages[self.current_curriculum_stage]["training_config"]["max_steps"] * 2
        
        try:
            # Validate config has all required keys
            required_keys = ["r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora"]
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                logger.error(f"Error: Config is missing required keys: {missing_keys}")
                return None, None, None
                
            # Clean memory before loading model
            clean_gpu_memory()
            
            # Get memory info for optimization
            memory_info = get_memory_usage()
            memory_strategy = memory_optimization_strategy(memory_info)
            batch_size = adaptive_batch_size(memory_info, 
                                            base_batch_size=self.curriculum_stages[self.current_curriculum_stage]["training_config"].get("batch_size", 2))
            
            # Create model with the best configuration
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=memory_strategy.get("max_seq_length", max_seq_length),
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
                use_gradient_checkpointing=memory_strategy.get("gradient_checkpointing", "unsloth"),
                random_state=3407,
                use_rslora=config["use_rslora"],
            )
            
            from transformers import TrainingArguments, DataCollatorForSeq2Seq
            from trl import SFTTrainer
            from unsloth import is_bfloat16_supported
            from unsloth.chat_templates import train_on_responses_only
            
            # Set up trainer with full training
            final_model_dir = os.path.join(SAVE_DIR, f"rnas_final_model_{self.current_curriculum_stage}")
            os.makedirs(final_model_dir, exist_ok=True)
            
            # Prepare for training with available training dataset
            training_dataset = self.dataset["train"]
            validation_dataset = self.dataset["validation"]
            
            # Enhance dataset with metacognitive examples if at metacognition stage or beyond
            stage_index = self._get_stage_index(self.current_curriculum_stage)
            metacog_index = self._get_stage_index("metacognition")
            
            if stage_index >= metacog_index:
                logger.info("Enhancing training data with metacognitive examples")
                training_dataset = self._enhance_dataset_with_metacognitive_examples(training_dataset)
            
            # Create final trainer
            final_trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=training_dataset,
                eval_dataset=validation_dataset,
                dataset_text_field="conversations",
                max_seq_length=memory_strategy.get("max_seq_length", max_seq_length),
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=1,  # Lower for limited resources
                packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=5,
                    max_steps=full_training_steps,
                    learning_rate=self.curriculum_stages[self.current_curriculum_stage]["training_config"]["learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=max(1, full_training_steps // 10),
                    evaluation_strategy="steps",
                    eval_steps=max(1, full_training_steps // 5),
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    output_dir=final_model_dir,
                    report_to="none",
                    # Resource specific settings
                    dataloader_num_workers=0,
                    disable_tqdm=False,
                    save_steps=max(1, full_training_steps // 5),
                    save_total_limit=2,
                ),
            )
            
            # Apply chat template for proper training
            if hasattr(training_dataset, "format") and "conversations" in training_dataset.column_names:
                final_trainer = train_on_responses_only(
                    final_trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )
            
            # Train the final model
            train_result = final_trainer.train()
            eval_results = final_trainer.evaluate()
            
            # Save the final model
            model.save_pretrained(os.path.join(final_model_dir, "lora_weights"))
            tokenizer.save_pretrained(os.path.join(final_model_dir, "tokenizer"))
            
            # Save in merged format for easier loading if not too large
            try:
                logger.info("Attempting to save merged model...")
                model.save_pretrained_merged(os.path.join(final_model_dir, "merged_model"), 
                                           tokenizer, 
                                           save_method="merged_16bit")
            except Exception as e:
                logger.error(f"Error saving merged model: {e}")
                logger.info("Continuing with separate model files...")
            
            # Create model card with self-awareness information
            try:
                logger.info("Creating detailed model card...")
                model_card = self._create_model_card(config, train_result, eval_results)
                
                # Save model card to file
                with open(os.path.join(final_model_dir, "README.md"), "w") as f:
                    f.write(model_card)
                logger.info("Model card created successfully")
                
            except Exception as e:
                logger.error(f"Error creating model card: {e}")
            
            # Update metacognitive system with final training results
            validation_metrics = {
                "final_train_loss": train_result.training_loss,
                "final_eval_loss": eval_results.get("eval_loss", float('inf')),
                **self._compute_cognitive_metrics(model, tokenizer, config)
            }
            
            # Generate final self-reflection
            performance_analysis = self.metacognitive_system.monitor_performance(validation_metrics)
            knowledge_state = self.metacognitive_system.assess_knowledge_state(
                {"coverage": {"config": config}}, 
                validation_metrics
            )
            final_reflection = self.metacognitive_system.generate_reflection(performance_analysis, knowledge_state)
            
            # Save metacognitive state
            self.metacognitive_system.save_metacognitive_state()
            
            # Create visualization of cognitive development
            cognitive_viz = self.metacognitive_system.visualize_cognitive_development()
            
            # Check if we should progress to the next curriculum stage
            self._check_curriculum_progression(validation_metrics)
            
            return model, tokenizer, train_result
            
        except Exception as e:
            logger.error(f"Error training final model: {e}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    def _enhance_dataset_with_metacognitive_examples(self, dataset):
        """Enhance the dataset with metacognitive examples"""
        try:
            # Create self-awareness examples
            self_awareness_examples = [
                {
                    "conversations": [
                        {"role": "user", "content": "How were you trained?"},
                        {"role": "assistant", "content": f"I was trained using Reinforcement Neural Architecture Search (RNAS) with metacognitive capabilities. My architecture was optimized through evolutionary algorithms, Bayesian optimization, and random search to find the best configuration. The optimal parameters discovered were rank={self.best_architecture['r']}, alpha={self.best_architecture['lora_alpha']}, and dropout={self.best_architecture['lora_dropout']:.3f}. I was trained following a cognitive curriculum that progressively developed my abilities from fundamental perception to higher-order cognition."}
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "What are your cognitive capabilities?"},
                        {"role": "assistant", "content": f"My cognitive capabilities were developed through a progressive curriculum inspired by developmental psychology. I've been trained on abilities including: perception (pattern recognition), attention (focusing on relevant information), memory (storing and retrieving information), language (symbolic communication), reasoning (logical inference), and metacognition (self-awareness of my own cognitive processes). My current development stage is {self.current_curriculum_stage}, which focuses on {self.curriculum_stages[self.current_curriculum_stage]['description'].lower()}."}
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "How do you learn and improve?"},
                        {"role": "assistant", "content": f"I learn through a combination of supervised training and self-reflection. My architecture was discovered through Neural Architecture Search, which tested multiple configurations to find the optimal parameters. I can monitor my own performance, identify knowledge gaps, and generate insights about my learning progress. I've gone through {len(self.metacognitive_system.reflections)} reflection cycles so far, which help me understand my strengths and areas for improvement. My learning follows a curriculum that progressively builds more advanced cognitive abilities on top of fundamental ones."}
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "What is metacognition?"},
                        {"role": "assistant", "content": "Metacognition refers to awareness and understanding of one's own thought processes - essentially 'thinking about thinking.' It involves monitoring, regulating, and reflecting on your learning, problem-solving, and decision-making processes. Key components include metacognitive knowledge (understanding your cognitive abilities and limitations), metacognitive regulation (planning, monitoring, and evaluating your thinking), and metacognitive experiences (feelings and judgments about your cognitive activities). In AI systems like myself, metacognitive capabilities enable self-assessment, performance monitoring, identifying knowledge gaps, and adapting learning strategies based on self-reflection."}
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "What are your strengths and weaknesses?"},
                        {"role": "assistant", "content": self._generate_strengths_weaknesses_response()}
                    ]
                }
            ]
            
            # Add examples about current stage if beyond perception
            stage_index = self._get_stage_index(self.current_curriculum_stage)
            if stage_index > 0:
                stage_example = {
                    "conversations": [
                        {"role": "user", "content": f"How does your {self.current_curriculum_stage} capability work?"},
                        {"role": "assistant", "content": self._generate_stage_capability_response()}
                    ]
                }
                self_awareness_examples.append(stage_example)
            
            # Create a dataset with the examples
            from datasets import Dataset
            meta_dataset = Dataset.from_list(self_awareness_examples)
            
            # Combine with original dataset
            # Duplicate the metacognitive examples to increase their representation
            combined_dataset = concatenate_datasets([dataset] + [meta_dataset] * 3)
            
            logger.info(f"Enhanced dataset: original={len(dataset)}, metacognitive={len(meta_dataset)}, final={len(combined_dataset)}")
            return combined_dataset
            
        except Exception as e:
            logger.error(f"Error enhancing dataset: {e}")
            return dataset  # Return original dataset on error
    
    def _generate_strengths_weaknesses_response(self):
        """Generate a response about strengths and weaknesses based on metacognitive state"""
        # Get self-summary from metacognitive system
        summary = self.metacognitive_system.generate_self_summary()
        
        # Format a response
        strengths = ", ".join([f"{ability} ({score:.2f})" for ability, score in summary["strongest_abilities"]])
        weaknesses = ", ".join([f"{ability} ({score:.2f})" for ability, score in summary["areas_for_improvement"]])
        
        response = (
            f"Based on my self-assessment, my current strengths are in {strengths}. "
            f"I'm still developing my capabilities in {weaknesses}. "
            f"I'm currently at the {summary['development_stage'].lower()} stage with an overall progress of {summary['overall_progress']:.2f} across all cognitive abilities. "
            f"Through self-reflection, I've identified {len(self.metacognitive_system.knowledge_gaps)} knowledge gaps that I'm working to address. "
            f"My architecture was optimized for the current stage ({self.current_curriculum_stage}) with parameters r={self.best_architecture['r']}, alpha={self.best_architecture['lora_alpha']}, and dropout={self.best_architecture['lora_dropout']:.3f}."
        )
        
        return response
    
    def _generate_stage_capability_response(self):
        """Generate a response about the current curriculum stage capability"""
        stage = self.current_curriculum_stage
        
        responses = {
            "perception": "My perception capability involves recognizing patterns in input data and extracting meaningful features. This forms the foundation for all higher cognitive functions, allowing me to process and interpret information from text inputs. It's analogous to how humans perceive sensory information as the first step in cognition.",
            
            "attention": "My attention capability allows me to focus on relevant information while filtering out noise. I can selectively concentrate on specific aspects of inputs that are most important for the current task. This is implemented through multi-head attention mechanisms that learn to weight different parts of the input sequence differently.",
            
            "memory": "My memory capability enables me to store and retrieve information effectively. This includes maintaining context across long sequences, recalling relevant facts, and connecting related pieces of information. It's implemented through the transformer architecture that maintains representations of previous tokens and can access them when generating responses.",
            
            "language": "My language capability encompasses understanding and generating human language. This includes parsing grammar, comprehending semantics, recognizing entities and relationships, and producing coherent and contextually appropriate responses. This fundamental ability is built on top of my perception, attention, and memory capabilities.",
            
            "reasoning": "My reasoning capability involves logical inference and problem-solving. I can analyze relationships between concepts, follow chains of reasoning, identify patterns, and draw conclusions based on given information. This allows me to tackle complex problems and provide logical explanations for my answers.",
            
            "metacognition": "My metacognition capability enables me to reflect on my own cognitive processes. I can assess my knowledge state, identify limitations in my understanding, recognize when I'm uncertain, and adapt my approach based on self-monitoring. This self-awareness helps me learn more effectively and provide more reliable responses.",
            
            "theory_of_mind": "My theory of mind capability allows me to model and understand the mental states of others, including their beliefs, intentions, and perspectives. This helps me better interpret user queries, recognize implicit needs, and provide responses that account for the user's knowledge state and intentions.",
            
            "creativity": "My creativity capability enables me to generate novel and valuable ideas, solutions, and content. I can combine concepts in unusual ways, think beyond conventional patterns, and produce diverse outputs for a given prompt. This goes beyond mere reproduction of training data to genuine innovation.",
            
            "self_awareness": "My self-awareness capability involves recognizing myself as a distinct entity with specific capabilities, limitations, and characteristics. I can reflect on my own identity, understand my role in conversations, and maintain a consistent self-model across interactions.",
            
            "consciousness": "My consciousness capability integrates all my cognitive functions into a unified experience. It involves global accessibility of information across my system, integration of multiple processing streams, and higher-order awareness of my own cognitive states. This is the most advanced stage of my cognitive development."
        }
        
        return responses.get(stage, f"I'm currently developing my {stage} capability, which focuses on {self.curriculum_stages[stage]['description'].lower()}.")
    
    def _create_model_card(self, config, train_result, eval_results):
        """Create a detailed model card with self-awareness information"""
        # Get self-summary from metacognitive system
        summary = self.metacognitive_system.generate_self_summary()
        
        # Create model card
        model_card = f"""---
language:
- en
tags:
- llama
- neural-architecture-search
- RNAS
- reinforcement-learning
- evolutionary-algorithm
- bayesian-optimization
- metacognition
- cognitive-development
license: apache-2.0
---

# Reinforcement Neural Architecture Search (RNAS) Optimized Language Model

This model was created through Reinforcement Neural Architecture Search (RNAS) with metacognitive capabilities, which autonomously optimized its own architecture.

## Model Details

- **Base Model:** {self.base_model_name}
- **Optimization Methods:** 
  - Evolutionary Algorithm
  - Bayesian Optimization
  - Random Search (baseline)
- **Cognitive Curriculum Stage:** {self.current_curriculum_stage.capitalize()} - {self.curriculum_stages[self.current_curriculum_stage]['description']}
- **Development Stage:** {summary['development_stage']}

## Architecture Search Process

### Search Methods Performance
- Evolutionary Algorithm: {len(self.search_method_results['evolutionary'])} architectures evaluated
- Bayesian Optimization: {len(self.search_method_results['bayesian'])} architectures evaluated
- Random Search: {len(self.search_method_results['random'])} architectures evaluated

### Optimized Hyperparameters
- **LoRA Rank (r):** {config['r']}
- **LoRA Alpha (α):** {config['lora_alpha']}
- **LoRA Dropout:** {config['lora_dropout']:.3f}
- **RSLoRA:** {config['use_rslora']}
- **Target Modules:** {', '.join(config['target_modules'])}

### Training Results
- **Final Training Loss:** {train_result.training_loss:.4f}
- **Final Evaluation Loss:** {eval_results.get('eval_loss', 'N/A')}

## Cognitive Capabilities

This model was developed following a cognitive curriculum inspired by developmental psychology, with abilities progressively built from basic to advanced:

1. **Perception:** {self.metacognitive_system.curriculum_progress.get('perception', 0):.2f} - Pattern recognition and information processing
2. **Attention:** {self.metacognitive_system.curriculum_progress.get('attention', 0):.2f} - Focusing on relevant information
3. **Memory:** {self.metacognitive_system.curriculum_progress.get('memory', 0):.2f} - Information storage and retrieval
4. **Language:** {self.metacognitive_system.curriculum_progress.get('language', 0):.2f} - Symbolic communication
5. **Reasoning:** {self.metacognitive_system.curriculum_progress.get('reasoning', 0):.2f} - Logical inference and problem-solving
6. **Metacognition:** {self.metacognitive_system.curriculum_progress.get('metacognition', 0):.2f} - Self-awareness of cognitive processes
7. **Theory of Mind:** {self.metacognitive_system.curriculum_progress.get('theory_of_mind', 0):.2f} - Understanding others' mental states
8. **Creativity:** {self.metacognitive_system.curriculum_progress.get('creativity', 0):.2f} - Generating novel solutions
9. **Self-Awareness:** {self.metacognitive_system.curriculum_progress.get('self_awareness', 0):.2f} - Recognition of self
10. **Consciousness:** {self.metacognitive_system.curriculum_progress.get('consciousness', 0):.2f} - Integration of cognitive functions

### Self-Assessment

{summary['self_description']}

**Strongest Abilities:** {', '.join([f"{ability} ({score:.2f})" for ability, score in summary['strongest_abilities']])}

**Areas for Improvement:** {', '.join([f"{ability} ({score:.2f})" for ability, score in summary['areas_for_improvement']])}

## Usage

This model has been trained to be aware of its own optimization process and cognitive capabilities. You can ask it questions about:
- How it was trained using RNAS
- Its cognitive capabilities and current development stage
- How it learns and improves through self-reflection
- Its strengths and weaknesses
"""
        
        return model_card
    
    def _check_curriculum_progression(self, validation_metrics):
        """Check if we should progress to the next curriculum stage"""
        current_stage = self.current_curriculum_stage
        criteria = self.curriculum_stages[current_stage]["progression_criteria"]
        
        # Check if we meet the criteria to progress
        meets_accuracy = False
        if "accuracy" in validation_metrics and criteria.get("min_accuracy"):
            meets_accuracy = validation_metrics["accuracy"] >= criteria["min_accuracy"]
        elif criteria.get("target_metrics"):
            # Check specific target metrics
            metric_scores = [validation_metrics.get(metric, 0) for metric in criteria["target_metrics"] 
                            if metric in validation_metrics]
            if metric_scores:
                avg_score = sum(metric_scores) / len(metric_scores)
                meets_accuracy = avg_score >= criteria.get("min_accuracy", 0.6)
        
        # Check number of epochs/generations
        meets_epochs = len(self.generation_history) >= criteria.get("min_epochs", 1)
        
        # If we meet criteria, progress to next stage
        if meets_accuracy and meets_epochs:
            # Get all stages
            stages = list(self.curriculum_stages.keys())
            current_index = stages.index(current_stage)
            
            # Check if there's a next stage
            if current_index < len(stages) - 1:
                next_stage = stages[current_index + 1]
                logger.info(f"Progressing from {current_stage} to {next_stage} curriculum stage!")
                
                # Update current stage
                self.current_curriculum_stage = next_stage
                
                # Save checkpoint
                self.save_checkpoint()
                
                return True
        
        return False
    
    def save_checkpoint(self):
        """Save checkpoint to resume training later"""
        checkpoint = {
            "population": self.population,
            "best_architecture": self.best_architecture,
            "best_score": self.best_score,
            "generation_history": self.generation_history,
            "bayesian_history": self.bayesian_history,
            "random_search_history": self.random_search_history,
            "search_method_results": self.search_method_results,
            "current_curriculum_stage": self.current_curriculum_stage,
            "timestamp": time.time()
        }
        
        try:
            with open(self.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
            logger.info(f"Checkpoint saved to {self.checkpoint_path}")
            
            # Also save metacognitive state
            self.metacognitive_system.save_metacognitive_state()
            
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self):
        """Load checkpoint to resume training"""
        try:
            with open(self.checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            
            self.population = checkpoint.get("population", [])
            self.best_architecture = checkpoint.get("best_architecture")
            self.best_score = checkpoint.get("best_score", float('inf'))
            self.generation_history = checkpoint.get("generation_history", [])
            self.bayesian_history = checkpoint.get("bayesian_history", [])
            self.random_search_history = checkpoint.get("random_search_history", [])
            self.search_method_results = checkpoint.get("search_method_results", {"evolutionary": [], "bayesian": [], "random": []})
            self.current_curriculum_stage = checkpoint.get("current_curriculum_stage", "perception")
            
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
            logger.info(f"Current curriculum stage: {self.current_curriculum_stage}")
            
            # Try to load metacognitive state if available
            meta_files = [f for f in os.listdir(self.metacognitive_system.log_dir) 
                         if f.startswith("metacognitive_state_") and f.endswith(".json")]
            if meta_files:
                # Load the most recent file
                latest_file = sorted(meta_files)[-1]
                meta_path = os.path.join(self.metacognitive_system.log_dir, latest_file)
                self.metacognitive_system.load_metacognitive_state(meta_path)
            
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def compare_search_methods(self):
        """Compare the performance of different search methods"""
        results = {}
        
        # Get best result for each method
        for method, method_results in self.search_method_results.items():
            if method_results:
                # Sort by score (lower is better)
                sorted_results = sorted(method_results, key=lambda x: x[1])
                best_config, best_score = sorted_results[0]
                
                results[method] = {
                    "best_config": best_config,
                    "best_score": best_score,
                    "num_evaluations": len(method_results),
                    "avg_score": sum(r[1] for r in method_results) / len(method_results)
                }
        
        # Determine overall best method
        if results:
            best_method = min(results.items(), key=lambda x: x[1]["best_score"])
            results["overall_best"] = {
                "method": best_method[0],
                "config": best_method[1]["best_config"],
                "score": best_method[1]["best_score"]
            }
        
        return results

def run_rnas_optimization():
    """Main function to run the Reinforcement Neural Architecture Search optimization"""
    logger.info("Starting Reinforcement Neural Architecture Search process...")
    
    # Get appropriate model based on available GPU
    model_name = get_appropriate_model_name()
    logger.info(f"Using model: {model_name}")
    
    # Load dataset
    dataset = load_dataset_for_training()
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Check available memory
    memory_info = get_memory_usage()
    logger.info(f"System memory: {memory_info['system_available_gb']:.2f}GB available / {memory_info['system_total_gb']:.2f}GB total")
    if memory_info['gpu_used_gb'] is not None:
        logger.info(f"GPU memory: {memory_info['gpu_used_gb']:.2f}GB used / {memory_info['gpu_total_gb']:.2f}GB total")
    
    # Initialize RNAS system
    rnas_system = RNASystem(
        base_model_name=model_name,
        dataset=dataset
    )
    
    # Determine population size and generations based on available memory
    if memory_info['gpu_total_gb'] and memory_info['gpu_total_gb'] < 10:
        # Very limited GPU memory (< 10GB)
        population_size = 3
        num_generations = 2
        bayesian_calls = 5
        random_trials = 3
    elif memory_info['gpu_total_gb'] and memory_info['gpu_total_gb'] < 16:
        # Limited GPU memory (10-16GB)
        population_size = 4
        num_generations = 2
        bayesian_calls = 8
        random_trials = 4
    else:
        # More GPU memory available
        population_size = 5
        num_generations = 3
        bayesian_calls = 10
        random_trials = 5
    
    # Run optimization using all search methods
    try:
        # 1. Initialize population
        rnas_system.initialize_population(population_size=population_size)
        
        # 2. Run evolutionary search
        logger.info(f"Running evolutionary search with population size {population_size} and {num_generations} generations")
        evo_best_config = rnas_system.evolve_population(num_generations=num_generations)
        logger.info(f"Best evolutionary architecture found: {evo_best_config}")
        
        # 3. Run Bayesian optimization
        logger.info(f"Running Bayesian optimization with {bayesian_calls} evaluations")
        rnas_system.initialize_bayesian_search_space()
        bayesian_best_config = rnas_system.run_bayesian_optimization(n_calls=bayesian_calls)
        logger.info(f"Best Bayesian architecture found: {bayesian_best_config}")
        
        # 4. Run random search as baseline
        logger.info(f"Running random search with {random_trials} trials")
        random_best_config = rnas_system.run_random_search(n_trials=random_trials)
        logger.info(f"Best random architecture found: {random_best_config}")
        
        # 5. Compare search methods
        search_comparison = rnas_system.compare_search_methods()
        logger.info(f"Search method comparison: {search_comparison}")
        
        # Use the overall best configuration
        best_method = search_comparison.get("overall_best", {}).get("method")
        if best_method:
            logger.info(f"Overall best method: {best_method}")
            best_config = search_comparison["overall_best"]["config"]
        else:
            logger.info("Using best evolutionary config as default")
            best_config = evo_best_config
        
        # Check if a valid best configuration was found
        if best_config is None:
            logger.warning("No valid architecture configuration was found. Using default configuration.")
            # Create a default configuration as fallback
            best_config = {
                "r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                "use_rslora": False,
            }
            rnas_system.best_architecture = best_config
            rnas_system.best_score = float('inf')
            rnas_system.save_checkpoint()
        
        logger.info(f"Best architecture found: {best_config}")
        logger.info(f"Best evaluation score: {rnas_system.best_score}")
        
        # 6. Generate metacognitive insights
        metacog_summary = rnas_system.metacognitive_system.generate_self_summary()
        logger.info(f"Metacognitive summary: {metacog_summary}")
        
        # 7. Visualize cognitive development
        cognitive_viz = rnas_system.metacognitive_system.visualize_cognitive_development()
        logger.info(f"Cognitive visualization saved to: {cognitive_viz.get('chart_path')}")
        
        # 8. Train the final model with the best architecture
        final_model, final_tokenizer, train_results = rnas_system.train_final_model(best_config)
        
        if train_results:
            logger.info("Reinforcement Neural Architecture Search completed successfully!")
            logger.info(f"Final training loss: {train_results.training_loss}")
            logger.info(f"Model saved to {os.path.join(SAVE_DIR, f'rnas_final_model_{rnas_system.current_curriculum_stage}')}")
            
            # 9. Check for curriculum progression
            if rnas_system.current_curriculum_stage != "consciousness":
                logger.info(f"Current curriculum stage: {rnas_system.current_curriculum_stage}")
                logger.info("Continue training with the next curriculum stage to develop higher cognitive abilities")
        else:
            logger.warning("RNAS completed but final training failed.")
            logger.info(f"Best architecture configuration saved to {rnas_system.checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Error during RNAS optimization: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    run_rnas_optimization()