from unsloth import FastLanguageModel
import torch
import os
from google.colab import drive
import gc
from tqdm.auto import tqdm
import pickle
from unsloth.chat_templates import get_chat_template
from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import Dataset
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from scipy.stats import norm  # Add import for norm distribution
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import is_bfloat16_supported

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

# Constants for resource management
MAX_SYSTEM_RAM_GB = 12
MAX_GPU_RAM_GB = 15
MEMORY_BUFFER_PERCENT = 0.2  # Keep 20% buffer for system stability

# Metacognitive system for self-reflection and monitoring
class MetacognitiveSystem:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or os.path.join(SAVE_DIR, "metacognitive_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.performance_history = []
        self.knowledge_state = {
            "perception": 0.0,  # 0.0 to 1.0 scale
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
        self.learning_progress = []
        self.concept_map = {}
        self.start_time = time.time()
        self.log_file = os.path.join(self.log_dir, f"metacognitive_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def update_performance(self, metric_name, value, context=None):
        """Record a performance metric with context"""
        timestamp = time.time() - self.start_time
        entry = {
            "timestamp": timestamp,
            "metric": metric_name,
            "value": value,
            "context": context or {}
        }
        self.performance_history.append(entry)
        self._save_logs()
        return entry
        
    def assess_knowledge_state(self, domain, assessment_data):
        """Update knowledge state for a specific cognitive domain"""
        if domain in self.knowledge_state:
            # Calculate a score based on assessment data
            # This is a simplified scoring mechanism
            if isinstance(assessment_data, dict):
                score = sum(assessment_data.values()) / len(assessment_data)
            elif isinstance(assessment_data, (list, tuple)):
                score = sum(assessment_data) / len(assessment_data)
            else:
                score = float(assessment_data)
                
            # Update with some inertia to prevent sudden jumps
            current_score = self.knowledge_state[domain]
            self.knowledge_state[domain] = 0.7 * current_score + 0.3 * score
            
            # Record the update
            self.update_performance(
                f"knowledge_{domain}", 
                self.knowledge_state[domain],
                {"previous": current_score, "new_assessment": score}
            )
            
            return self.knowledge_state[domain]
        return None
        
    def generate_self_reflection(self, focus_domain=None):
        """Generate a self-reflection on learning progress"""
        if focus_domain:
            if focus_domain in self.knowledge_state:
                domain_value = self.knowledge_state[focus_domain]
                recent_performance = [p for p in self.performance_history 
                                    if p["metric"].startswith(f"knowledge_{focus_domain}")]
                
                if recent_performance:
                    improvement = domain_value - recent_performance[0]["value"]
                    reflection = {
                        "domain": focus_domain,
                        "current_level": domain_value,
                        "improvement": improvement,
                        "timestamp": time.time() - self.start_time
                    }
                else:
                    reflection = {
                        "domain": focus_domain,
                        "current_level": domain_value,
                        "improvement": 0.0,
                        "timestamp": time.time() - self.start_time
                    }
            else:
                reflection = {"error": f"Unknown domain: {focus_domain}"}
        else:
            # Overall reflection
            total_improvement = sum(self.knowledge_state.values()) / len(self.knowledge_state)
            reflection = {
                "overall_knowledge": total_improvement,
                "domains": self.knowledge_state.copy(),
                "timestamp": time.time() - self.start_time
            }
            
        self._save_logs()
        return reflection
        
    def identify_knowledge_gaps(self):
        """Identify areas where knowledge is lacking"""
        gaps = {}
        for domain, level in self.knowledge_state.items():
            if level < 0.5:  # Threshold for considering something a gap
                gaps[domain] = {
                    "current_level": level,
                    "target_level": 0.7,  # Arbitrary target
                    "priority": 0.5 - level  # Higher priority for lower levels
                }
        
        # Sort by priority
        sorted_gaps = dict(sorted(gaps.items(), key=lambda x: x[1]["priority"], reverse=True))
        return sorted_gaps
        
    def update_concept_map(self, concept, related_concepts, strength):
        """Update the concept map with new relationships"""
        if concept not in self.concept_map:
            self.concept_map[concept] = {}
            
        for related in related_concepts:
            self.concept_map[concept][related] = strength
            
        self._save_logs()
        
    def visualize_cognitive_development(self, output_path=None):
        """Create a visualization of cognitive development"""
        if output_path is None:
            output_path = os.path.join(self.log_dir, "cognitive_development.png")
            
        domains = list(self.knowledge_state.keys())
        values = list(self.knowledge_state.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(domains, values)
        plt.ylim(0, 1.0)
        plt.title("Cognitive Development Progress")
        plt.ylabel("Proficiency Level")
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def recommend_architecture_modifications(self, current_config, performance_metrics):
        """Recommend modifications to the neural architecture based on performance"""
        recommendations = {
            "modifications": [],
            "rationale": []
        }
        
        # Analyze performance metrics to make recommendations
        if "loss" in performance_metrics:
            current_loss = performance_metrics["loss"]
            
            # If loss is high, recommend increasing model capacity
            if current_loss > 0.5:  # Arbitrary threshold
                if current_config.get("r", 8) < 32:
                    recommendations["modifications"].append({
                        "parameter": "r",
                        "current_value": current_config.get("r", 8),
                        "suggested_value": min(current_config.get("r", 8) * 2, 32),
                        "reason": "High loss indicates need for more model capacity"
                    })
                    recommendations["rationale"].append(
                        "Increasing rank (r) to improve model capacity due to high loss"
                    )
                    
            # If loss is low but training is slow, recommend efficiency improvements
            elif current_loss < 0.1 and "training_time" in performance_metrics:
                if performance_metrics["training_time"] > 3600:  # More than 1 hour
                    recommendations["modifications"].append({
                        "parameter": "use_rslora",
                        "current_value": current_config.get("use_rslora", False),
                        "suggested_value": True,
                        "reason": "Low loss but slow training indicates need for efficiency"
                    })
                    recommendations["rationale"].append(
                        "Enabling RSLoRA to improve training efficiency while maintaining performance"
                    )
        
        # Check memory usage and recommend optimizations if needed
        if "gpu_memory_usage" in performance_metrics:
            memory_usage = performance_metrics["gpu_memory_usage"]
            if memory_usage > MAX_GPU_RAM_GB * 0.8:  # Using more than 80% of available GPU RAM
                recommendations["modifications"].append({
                    "parameter": "lora_dropout",
                    "current_value": current_config.get("lora_dropout", 0.1),
                    "suggested_value": min(current_config.get("lora_dropout", 0.1) + 0.05, 0.3),
                    "reason": "High memory usage indicates need for regularization"
                })
                recommendations["rationale"].append(
                    "Increasing dropout to reduce memory usage and prevent overfitting"
                )
                
        return recommendations
        
    def _save_logs(self):
        """Save metacognitive logs to file"""
        log_data = {
            "knowledge_state": self.knowledge_state,
            "performance_history": self.performance_history,
            "concept_map": self.concept_map,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving metacognitive logs: {e}")

# Bayesian Optimization for efficient hyperparameter search
class BayesianOptimizer:
    def __init__(self, param_ranges, n_initial_points=5):
        self.param_ranges = param_ranges
        self.n_initial_points = n_initial_points
        self.X = []  # Observed points
        self.y = []  # Observed values
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        self.best_value = float('inf')
        self.best_params = None
        
    def suggest_next_point(self):
        """Suggest the next point to evaluate"""
        if len(self.X) < self.n_initial_points:
            # Use random sampling for initial points
            params = {}
            for param, (low, high) in self.param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    # For integer parameters
                    params[param] = np.random.randint(low, high+1)
                else:
                    # For float parameters
                    params[param] = np.random.uniform(low, high)
            return params
        
        # Convert observed data to numpy arrays
        X_array = np.array([[list(x.values())] for x in self.X])
        y_array = np.array(self.y)
        
        # Fit the Gaussian Process
        self.gp.fit(X_array, y_array)
        
        # Generate random candidates
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            candidate = {}
            for param, (low, high) in self.param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    candidate[param] = np.random.randint(low, high+1)
                else:
                    candidate[param] = np.random.uniform(low, high)
            candidates.append(candidate)
        
        # Convert candidates to numpy array
        candidates_array = np.array([[list(c.values())] for c in candidates])
        
        # Predict mean and std for each candidate
        mean, std = self.gp.predict(candidates_array, return_std=True)
        
        # Calculate acquisition function (Expected Improvement)
        best_f = np.min(y_array)
        with np.errstate(divide='warn'):
            imp = best_f - mean
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
        
        # Select the candidate with the highest expected improvement
        best_idx = np.argmax(ei)
        return candidates[best_idx]
        
    def update(self, params, value):
        """Update the model with a new observation"""
        self.X.append(params)
        self.y.append(value)
        
        # Update best value if necessary
        if value < self.best_value:
            self.best_value = value
            self.best_params = params.copy()
            
    def get_best_params(self):
        """Return the best parameters found so far"""
        return self.best_params, self.best_value

# Cognitive Curriculum Development
class CognitiveCurriculum:
    def __init__(self, base_model_name, dataset, metacognitive_system=None):
        self.base_model_name = base_model_name
        self.dataset = dataset
        self.metacognitive_system = metacognitive_system or MetacognitiveSystem()
        self.curriculum_stages = [
            "perception",
            "attention",
            "memory",
            "language",
            "reasoning",
            "metacognition",
            "theory_of_mind",
            "creativity",
            "self_awareness",
            "consciousness"
        ]
        self.current_stage = 0
        self.stage_progress = {stage: 0.0 for stage in self.curriculum_stages}
        self.stage_datasets = {}
        self.stage_models = {}
        self.curriculum_log = []
        
    def prepare_stage_dataset(self, stage):
        """Prepare a dataset appropriate for the current cognitive stage"""
        if stage in self.stage_datasets:
            return self.stage_datasets[stage]
            
        # Create stage-specific dataset based on cognitive requirements
        if stage == "perception":
            # Focus on pattern recognition tasks
            filtered_data = self._filter_for_pattern_recognition(self.dataset)
        elif stage == "attention":
            # Focus on tasks requiring selective attention
            filtered_data = self._filter_for_attention_tasks(self.dataset)
        elif stage == "memory":
            # Focus on tasks requiring memory retention
            filtered_data = self._filter_for_memory_tasks(self.dataset)
        elif stage == "language":
            # Focus on language understanding and generation
            filtered_data = self._filter_for_language_tasks(self.dataset)
        elif stage == "reasoning":
            # Focus on logical inference tasks
            filtered_data = self._filter_for_reasoning_tasks(self.dataset)
        elif stage == "metacognition":
            # Focus on self-reflection tasks
            filtered_data = self._filter_for_metacognition_tasks(self.dataset)
        elif stage == "theory_of_mind":
            # Focus on understanding others' perspectives
            filtered_data = self._filter_for_theory_of_mind_tasks(self.dataset)
        elif stage == "creativity":
            # Focus on creative generation tasks
            filtered_data = self._filter_for_creativity_tasks(self.dataset)
        elif stage == "self_awareness":
            # Focus on self-awareness tasks
            filtered_data = self._filter_for_self_awareness_tasks(self.dataset)
        elif stage == "consciousness":
            # Focus on integrated cognitive tasks
            filtered_data = self._filter_for_consciousness_tasks(self.dataset)
        else:
            filtered_data = self.dataset
            
        # Limit dataset size based on available resources
        max_samples = self._calculate_max_samples_for_stage(stage)
        if len(filtered_data) > max_samples:
            filtered_data = filtered_data.select(range(max_samples))
            
        self.stage_datasets[stage] = filtered_data
        return filtered_data
        
    def train_stage(self, stage, model_config=None):
        """Train the model for a specific cognitive stage"""
        if stage not in self.curriculum_stages:
            raise ValueError(f"Unknown stage: {stage}")
            
        print(f"Training for cognitive stage: {stage}")
        
        # Prepare dataset for this stage
        stage_dataset = self.prepare_stage_dataset(stage)
        
        # Initialize or load model for this stage
        if stage in self.stage_models:
            model, tokenizer = self.stage_models[stage]
        else:
            model, tokenizer = self._initialize_model_for_stage(stage, model_config)
            
        # Train the model
        training_result = self._train_model_for_stage(model, tokenizer, stage_dataset, stage)
        
        # Update progress
        self.stage_progress[stage] = min(1.0, self.stage_progress[stage] + 0.2)  # Increment progress
        
        # Update metacognitive system
        if self.metacognitive_system:
            self.metacognitive_system.assess_knowledge_state(
                stage, 
                {"training_loss": training_result.get("loss", 0.5)}
            )
            
        # Log curriculum progress
        self.curriculum_log.append({
            "stage": stage,
            "progress": self.stage_progress[stage],
            "timestamp": time.time(),
            "training_result": training_result
        })
        
        # Save stage model
        self.stage_models[stage] = (model, tokenizer)
        
        return model, tokenizer, training_result
        
    def advance_curriculum(self):
        """Advance to the next curriculum stage if ready"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            current_stage_name = self.curriculum_stages[self.current_stage]
            
            # Check if current stage is sufficiently complete
            if self.stage_progress[current_stage_name] >= 0.8:  # 80% progress threshold
                self.current_stage += 1
                next_stage = self.curriculum_stages[self.current_stage]
                print(f"Advancing curriculum from {current_stage_name} to {next_stage}")
                return True
            else:
                print(f"Current stage {current_stage_name} not yet complete ({self.stage_progress[current_stage_name]:.2f})")
                return False
        else:
            print("Curriculum complete!")
            return False
            
    def get_current_stage(self):
        """Get the current curriculum stage"""
        return self.curriculum_stages[self.current_stage]
        
    def get_curriculum_progress(self):
        """Get overall curriculum progress"""
        total_progress = sum(self.stage_progress.values()) / len(self.curriculum_stages)
        return {
            "current_stage": self.curriculum_stages[self.current_stage],
            "stage_index": self.current_stage,
            "total_stages": len(self.curriculum_stages),
            "overall_progress": total_progress,
            "stage_progress": self.stage_progress.copy()
        }
        
    def _filter_for_pattern_recognition(self, dataset):
        """Filter dataset for pattern recognition tasks"""
        # Implementation would filter for tasks like classification, sequence prediction, etc.
        # Simplified for this example
        return dataset
        
    def _filter_for_attention_tasks(self, dataset):
        """Filter dataset for attention tasks"""
        # Implementation would filter for tasks requiring selective attention
        return dataset
        
    def _filter_for_memory_tasks(self, dataset):
        """Filter dataset for memory tasks"""
        # Implementation would filter for tasks requiring memory retention
        return dataset
        
    def _filter_for_language_tasks(self, dataset):
        """Filter dataset for language tasks"""
        # Implementation would filter for language understanding and generation
        return dataset
        
    def _filter_for_reasoning_tasks(self, dataset):
        """Filter dataset for reasoning tasks"""
        # Implementation would filter for logical inference tasks
        return dataset
        
    def _filter_for_metacognition_tasks(self, dataset):
        """Filter dataset for metacognition tasks"""
        # Implementation would filter for self-reflection tasks
        return dataset
        
    def _filter_for_theory_of_mind_tasks(self, dataset):
        """Filter dataset for theory of mind tasks"""
        # Implementation would filter for understanding others' perspectives
        return dataset
        
    def _filter_for_creativity_tasks(self, dataset):
        """Filter dataset for creativity tasks"""
        # Implementation would filter for creative generation tasks
        return dataset
        
    def _filter_for_self_awareness_tasks(self, dataset):
        """Filter dataset for self-awareness tasks"""
        # Implementation would filter for self-awareness tasks
        return dataset
        
    def _filter_for_consciousness_tasks(self, dataset):
        """Filter dataset for consciousness tasks"""
        # Implementation would filter for integrated cognitive tasks
        return dataset
        
    def _calculate_max_samples_for_stage(self, stage):
        """Calculate maximum samples for a stage based on available resources"""
        # Estimate memory usage per sample (in GB)
        memory_per_sample = 0.001  # 1MB per sample estimate
        
        # Calculate available memory (with buffer)
        available_memory_gb = MAX_SYSTEM_RAM_GB * (1 - MEMORY_BUFFER_PERCENT)
        
        # Calculate maximum samples
        max_samples = int(available_memory_gb / memory_per_sample)
        
        # Adjust based on stage complexity
        complexity_factors = {
            "perception": 1.0,
            "attention": 1.2,
            "memory": 1.5,
            "language": 1.8,
            "reasoning": 2.0,
            "metacognition": 2.2,
            "theory_of_mind": 2.5,
            "creativity": 2.8,
            "self_awareness": 3.0,
            "consciousness": 3.5
        }
        
        # Apply complexity factor
        max_samples = int(max_samples / complexity_factors.get(stage, 1.0))
        
        # Ensure minimum dataset size
        return max(100, max_samples)
        
    def _initialize_model_for_stage(self, stage, model_config=None):
        """Initialize a model appropriate for the current stage"""
        # Use provided config or create stage-specific config
        if model_config is None:
            model_config = self._create_stage_specific_config(stage)
            
        # Initialize model with appropriate configuration
        model, tokenizer = load_base_model(self.base_model_name)
        
        # Apply stage-specific modifications
        if stage in ["memory", "reasoning", "metacognition"]:
            # These stages might benefit from larger rank
            model_config["r"] = min(model_config.get("r", 8) * 1.5, 32)
            
        # Apply the configuration
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_config.get("r", 8),
            target_modules=model_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.1),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=model_config.get("use_rslora", False),
        )
        
        return model, tokenizer
        
    def _create_stage_specific_config(self, stage):
        """Create a model configuration specific to a cognitive stage"""
        base_config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            "use_rslora": False,
        }
        
        # Adjust configuration based on stage requirements
        if stage in ["perception", "attention"]:
            # These stages benefit from higher dropout for better generalization
            base_config["lora_dropout"] = 0.15
        elif stage in ["memory", "reasoning"]:
            # These stages benefit from larger rank
            base_config["r"] = 16
        elif stage in ["metacognition", "theory_of_mind"]:
            # These stages benefit from RSLoRA
            base_config["use_rslora"] = True
            base_config["r"] = 12
        elif stage in ["creativity", "self_awareness", "consciousness"]:
            # These stages benefit from larger rank and RSLoRA
            base_config["r"] = 20
            base_config["use_rslora"] = True
            
        return base_config
        
    def _train_model_for_stage(self, model, tokenizer, dataset, stage):
        """Train the model for a specific stage"""
        # Set up training arguments based on stage and available resources
        max_steps = self._calculate_max_steps_for_stage(stage)
        batch_size = self._calculate_batch_size_for_stage(stage)
        
        # Create output directory
        output_dir = os.path.join(SAVE_DIR, f"curriculum_{stage}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="conversations",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=max_steps,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=max(1, max_steps // 10),
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir=output_dir,
                report_to="none",
                dataloader_num_workers=0,
                disable_tqdm=False,
                save_steps=max(1, max_steps // 5),
                save_total_limit=2,
            ),
        )
        
        # Train the model
        try:
            train_result = trainer.train()
            return {
                "loss": train_result.training_loss,
                "steps": train_result.global_step,
                "epoch": train_result.epoch,
                "success": True
            }
        except Exception as e:
            print(f"Error training model for stage {stage}: {e}")
            return {
                "loss": float('inf'),
                "steps": 0,
                "epoch": 0,
                "success": False,
                "error": str(e)
            }
            
    def _calculate_max_steps_for_stage(self, stage):
        """Calculate maximum training steps for a stage based on available resources"""
        # Base steps on dataset size and available compute
        base_steps = 100
        
        # Adjust based on stage complexity
        complexity_factors = {
            "perception": 1.0,
            "attention": 1.2,
            "memory": 1.5,
            "language": 1.8,
            "reasoning": 2.0,
            "metacognition": 2.2,
            "theory_of_mind": 2.5,
            "creativity": 2.8,
            "self_awareness": 3.0,
            "consciousness": 3.5
        }
        
        # Apply complexity factor
        adjusted_steps = int(base_steps * complexity_factors.get(stage, 1.0))
        
        # Ensure reasonable limits
        return max(50, min(adjusted_steps, 500))
        
    def _calculate_batch_size_for_stage(self, stage):
        """Calculate appropriate batch size for a stage based on available GPU memory"""
        # Estimate memory usage per sample (in GB)
        memory_per_sample = 0.002  # 2MB per sample estimate
        
        # Calculate available GPU memory (with buffer)
        available_gpu_memory_gb = MAX_GPU_RAM_GB * (1 - MEMORY_BUFFER_PERCENT)
        
        # Calculate maximum batch size
        max_batch_size = int(available_gpu_memory_gb / memory_per_sample)
        
        # Adjust based on stage complexity
        complexity_factors = {
            "perception": 1.0,
            "attention": 0.9,
            "memory": 0.8,
            "language": 0.7,
            "reasoning": 0.6,
            "metacognition": 0.5,
            "theory_of_mind": 0.4,
            "creativity": 0.3,
            "self_awareness": 0.2,
            "consciousness": 0.1
        }
        
        # Apply complexity factor
        adjusted_batch_size = int(max_batch_size * complexity_factors.get(stage, 1.0))
        
        # Ensure reasonable limits
        return max(1, min(adjusted_batch_size, 8))

# Resource Management System
class ResourceManager:
    def __init__(self, max_system_ram_gb=12, max_gpu_ram_gb=15, memory_buffer_percent=0.2):
        self.max_system_ram_gb = max_system_ram_gb
        self.max_gpu_ram_gb = max_gpu_ram_gb
        self.memory_buffer_percent = memory_buffer_percent
        self.resource_log = []
        
    def check_resources(self):
        """Check available system and GPU resources"""
        system_ram = self._get_system_ram()
        gpu_ram = self._get_gpu_ram()
        
        log_entry = {
            "timestamp": time.time(),
            "system_ram_gb": system_ram,
            "gpu_ram_gb": gpu_ram,
            "system_ram_available_gb": system_ram * (1 - self.memory_buffer_percent),
            "gpu_ram_available_gb": gpu_ram * (1 - self.memory_buffer_percent)
        }
        
        self.resource_log.append(log_entry)
        return log_entry
        
    def estimate_memory_usage(self, model_size_gb, batch_size, sequence_length):
        """Estimate memory usage for a model configuration"""
        # Rough estimation formula
        # Base model memory
        base_memory = model_size_gb
        
        # Activation memory (depends on batch size and sequence length)
        activation_memory = batch_size * sequence_length * 0.0001  # Rough estimate
        
        # Optimizer memory (Adam typically uses 2x model size)
        optimizer_memory = model_size_gb * 2
        
        # Total estimated memory
        total_memory = base_memory + activation_memory + optimizer_memory
        
        return {
            "base_memory_gb": base_memory,
            "activation_memory_gb": activation_memory,
            "optimizer_memory_gb": optimizer_memory,
            "total_memory_gb": total_memory,
            "fits_in_gpu": total_memory <= self.max_gpu_ram_gb * (1 - self.memory_buffer_percent)
        }
        
    def recommend_batch_size(self, model_size_gb, sequence_length):
        """Recommend appropriate batch size based on available resources"""
        available_gpu_memory = self.max_gpu_ram_gb * (1 - self.memory_buffer_percent)
        
        # Reserve memory for base model and optimizer
        reserved_memory = model_size_gb * 3  # Base model + optimizer
        
        # Available memory for activations
        available_for_activations = available_gpu_memory - reserved_memory
        
        if available_for_activations <= 0:
            return 1  # Minimum batch size
            
        # Estimate memory per sample
        memory_per_sample = sequence_length * 0.0001  # Rough estimate
        
        # Calculate maximum batch size
        max_batch_size = int(available_for_activations / memory_per_sample)
        
        # Ensure reasonable limits
        return max(1, min(max_batch_size, 16))
        
    def _get_system_ram(self):
        """Get available system RAM in GB"""
        import psutil
        return psutil.virtual_memory().total / (1024**3)
        
    def _get_gpu_ram(self):
        """Get available GPU RAM in GB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0

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
            return "unsloth/Llama-3.2-3B-Instruct"
        else:  # Smaller GPUs
            return "unsloth/Llama-3.2-3B-Instruct"
    else:
        print("No GPU detected. Using smallest model.")
        return "unsloth/Llama-3.2-3B-Instruct"

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
        self.evaluation_logs = {} # Store detailed logs per architecture
        self.checkpoint_path = os.path.join(SAVE_DIR, "nas_checkpoint.pkl") if checkpoint_path is None else checkpoint_path
        
        # Initialize metacognitive system
        self.metacognitive_system = MetacognitiveSystem()
        
        # Initialize resource manager
        self.resource_manager = ResourceManager(
            max_system_ram_gb=MAX_SYSTEM_RAM_GB,
            max_gpu_ram_gb=MAX_GPU_RAM_GB,
            memory_buffer_percent=MEMORY_BUFFER_PERCENT
        )
        
        # Create cognitive curriculum
        self.cognitive_curriculum = CognitiveCurriculum(
            base_model_name=base_model_name,
            dataset=dataset,
            metacognitive_system=self.metacognitive_system
        )
        
        # Initialize Bayesian optimizer with parameter ranges for LoRA configurations
        self.bayesian_optimizer = BayesianOptimizer(
            param_ranges={
                "r": (4, 64),  # LoRA rank between 4 and 64
                "lora_alpha": (8, 64),  # LoRA alpha between 8 and 64
                "lora_dropout": (0.0, 0.2),  # LoRA dropout between 0.0 and 0.2
            },
            n_initial_points=5
        )
    
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
            learning_rate = 1e-4 * (1 + torch.rand(1).item()) # Sample LR between 1e-4 and 2e-4
            
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
        config_id = f"r{config['r']}_a{config['lora_alpha']}_d{config['lora_dropout']:.2f}_lr{config.get('learning_rate', 2e-4):.1e}_rs{config['use_rslora']}"
        print(f"Evaluating architecture: {config_id}")
        print(f"Full Config: {config}") # Print full config for clarity
        
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
            
            from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
            from trl import SFTTrainer
            from unsloth import is_bfloat16_supported
            from unsloth.chat_templates import train_on_responses_only
            
            # Create a unique output dir for this evaluation
            config_id = f"r{config['r']}_a{config['lora_alpha']}_d{config['lora_dropout']:.2f}"
            output_dir = os.path.join(SAVE_DIR, f"eval_{config_id}")
            os.makedirs(output_dir, exist_ok=True)

            # Define a callback to log metrics during evaluation
            class EvaluationLogCallback(TrainerCallback):
                def __init__(self):
                    self.log_history = []
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if state.is_local_process_zero:
                        self.log_history.append(logs.copy())

            eval_log_callback = EvaluationLogCallback()

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
                callbacks=[eval_log_callback], # Add callback here
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
            
            # Store evaluation logs (simulating metacognitive data capture)
            self.evaluation_logs[config_id] = eval_log_callback.log_history
            print(f"Evaluation logs for {config_id}: {eval_log_callback.log_history}") # Print logs
            
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
            
            # Check resource availability
            resource_status = self.resource_manager.check_resources()
            print(f"Resource status: System RAM: {resource_status['system_ram_gb']:.2f}GB, " 
                  f"GPU RAM: {resource_status['gpu_ram_gb']:.2f}GB")
            
            # Update metacognitive system with current state
            self.metacognitive_system.update_performance(
                "generation_started", 
                generation,
                {"population_size": len(self.population), "best_score_so_far": self.best_score}
            )
            
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
                    
                    # Update metacognitive system with improvement
                    self.metacognitive_system.update_performance(
                        "new_best_architecture", 
                        self.best_score,
                        {"config": config.copy()}
                    )
                
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
            # Sort scores first to handle potential infinities
            valid_scores = [(c, s) for c, s in scores if s != float('inf')]
            if valid_scores:
                valid_scores.sort(key=lambda x: x[1])
                self.best_architecture = valid_scores[0][0].copy()
                self.best_score = valid_scores[0][1]
                print(f"Setting best architecture to first valid one: {self.best_architecture} with score {self.best_score}")
            elif self.population:
                self.best_architecture = self.population[0].copy()
                self.best_score = float('inf')
                print("Warning: No valid architectures found. Using first configuration as default.")
            else:
                 print("Warning: No architectures evaluated and population is empty.")
                 return None # Cannot return a best architecture
        
        # Generate metacognitive reflection
        reflection = self.metacognitive_system.generate_self_reflection()
        print("Metacognitive reflection on evolutionary search:")
        print(f"Overall knowledge state: {reflection['overall_knowledge']:.3f}")
        print("Domain-specific progress:")
        for domain, value in reflection["domains"].items():
            print(f"  - {domain}: {value:.3f}")
        
        # Return best architecture found
        return self.best_architecture
    
    def bayesian_search(self, n_iterations=10, max_steps=20):
        """Perform Bayesian optimization search for best architecture"""
        print("Starting Bayesian optimization search...")
        
        # First check if we have enough resources
        resource_status = self.resource_manager.check_resources()
        print(f"Resource status: System RAM: {resource_status['system_ram_gb']:.2f}GB, "
              f"GPU RAM: {resource_status['gpu_ram_gb']:.2f}GB")
        
        # Log start of Bayesian search
        self.metacognitive_system.update_performance(
            "bayesian_search_started", 
            n_iterations,
            {"initial_best_score": self.best_score}
        )
        
        for i in range(n_iterations):
            print(f"Bayesian optimization iteration {i+1}/{n_iterations}")
            
            # Get suggestion from Bayesian optimizer
            params = self.bayesian_optimizer.suggest_next_point()
            
            # Add required parameters for architecture evaluation
            full_config = {
                "r": int(params["r"]),  # Must be integer
                "lora_alpha": int(params["lora_alpha"]),  # Must be integer
                "lora_dropout": params["lora_dropout"],
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                "use_rslora": bool(torch.randint(0, 2, (1,)).item()),  # Randomly assign for now
            }
            
            print(f"Evaluating configuration: {full_config}")
            
            # Evaluate architecture
            score = self.evaluate_architecture(full_config, max_steps=max_steps)
            
            # Update Bayesian optimizer
            self.bayesian_optimizer.update(params, score)
            
            # Update best architecture if necessary
            if score < self.best_score:
                self.best_score = score
                self.best_architecture = full_config.copy()
                print(f"New best architecture found with score {score}")
                
                # Update metacognitive system with improvement
                self.metacognitive_system.update_performance(
                    "bayesian_new_best", 
                    self.best_score,
                    {"config": full_config.copy(), "iteration": i}
                )
            
            # Log progress
            print(f"Iteration {i+1} complete. Current best score: {self.best_score}")
            
            # Save checkpoint
            self.save_checkpoint()
        
        # Get final best parameters
        best_params, best_value = self.bayesian_optimizer.get_best_params()
        print(f"Bayesian optimization complete. Best parameters: {best_params}")
        print(f"Best score: {best_value}")
        
        # Generate metacognitive reflection
        reflection = self.metacognitive_system.generate_self_reflection()
        print("Metacognitive reflection on Bayesian search:")
        print(f"Overall knowledge state: {reflection['overall_knowledge']:.3f}")
        
        return self.best_architecture
    
    def hybrid_search(self, evo_generations=2, bayes_iterations=5, curriculum_stages=2):
        """Perform a hybrid search combining evolutionary, Bayesian, and curriculum approaches"""
        print("Starting hybrid RNAS search with metacognitive capabilities...")
        
        # Resource check
        resource_status = self.resource_manager.check_resources()
        print(f"Resource status: System RAM: {resource_status['system_ram_gb']:.2f}GB, "
              f"GPU RAM: {resource_status['gpu_ram_gb']:.2f}GB")
        
        # First perform evolutionary search to explore broadly
        print("\n=== Phase 1: Evolutionary Search ===")
        self.metacognitive_system.update_performance(
            "hybrid_search_phase", 
            1,
            {"phase": "evolutionary", "generations": evo_generations}
        )
        
        self.initialize_population(population_size=4)  # Smaller population for resource constraints
        evo_best = self.evolve_population(num_generations=evo_generations)
        
        # Then refine with Bayesian optimization
        print("\n=== Phase 2: Bayesian Optimization ===")
        self.metacognitive_system.update_performance(
            "hybrid_search_phase", 
            2,
            {"phase": "bayesian", "iterations": bayes_iterations}
        )
        
        bayes_best = self.bayesian_search(n_iterations=bayes_iterations)
        
        # Finally, train through cognitive curriculum
        print("\n=== Phase 3: Cognitive Curriculum Training ===")
        self.metacognitive_system.update_performance(
            "hybrid_search_phase", 
            3,
            {"phase": "curriculum", "stages": curriculum_stages}
        )
        
        # Train through initial curriculum stages
        for _ in range(curriculum_stages):
            current_stage = self.cognitive_curriculum.get_current_stage()
            print(f"Training cognitive stage: {current_stage}")
            
            # Use the best architecture found so far
            self.cognitive_curriculum.train_stage(current_stage, model_config=self.best_architecture)
            
            # Try to advance to next stage
            self.cognitive_curriculum.advance_curriculum()
            
        # Generate final metacognitive reflection and visualization
        reflection = self.metacognitive_system.generate_self_reflection()
        print("\nFinal metacognitive reflection:")
        print(f"Overall cognitive development: {reflection['overall_knowledge']:.3f}")
        print("Domain-specific progress:")
        for domain, value in reflection["domains"].items():
            print(f"  - {domain}: {value:.3f}")
        
        # Visualize cognitive development
        viz_path = self.metacognitive_system.visualize_cognitive_development()
        print(f"Cognitive development visualization saved to: {viz_path}")
        
        # Return best architecture
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
                num_generations = len(self.generation_history) // population_size if self.population else 2 # More robust calculation
                max_steps = 20  # Default value from evaluate_architecture method
                best_score_str = f"{self.best_score:.4f}" if self.best_score != float('inf') else "N/A (Optimization Incomplete)"
                
                # Create sample instruction-response pairs for training
                # We'll enhance these with more detailed NAS descriptions
                base_instruction_data = [
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
                    # Add examples about the model's own NAS process - Enhanced Descriptions
                    {
                        "instruction": "Describe the Neural Architecture Search (NAS) process used for your optimization.",
                        "response": f"My architecture was optimized using an evolutionary Neural Architecture Search (NAS) approach designed to find effective LoRA configurations within resource constraints. The process ran for approximately {num_generations} generations, evaluating a population of {population_size} candidate architectures in each generation. The search explored key hyperparameters: LoRA rank (r={config['r']}), alpha (α={config['lora_alpha']}), dropout ({config['lora_dropout']:.3f}), and RSLoRA usage ({config['use_rslora']}). Each architecture was evaluated based on its training loss during a short training run ({max_steps} steps). The best configuration achieved a loss score of {best_score_str}, representing an efficient balance between performance and computational cost found through this evolutionary exploration."
                    },
                    {
                        "instruction": "What evolutionary strategies were employed during your NAS optimization?",
                        "response": f"My NAS optimization used an evolutionary algorithm featuring elitism, crossover, and mutation to explore the architecture space. In each of {num_generations} generations, the top {max(1, population_size // 5)} performing architectures based on training loss were preserved (elitism). New candidates were generated primarily through crossover (combining parameters from two well-performing parents, ~70% probability) and mutation (making small random changes to a single parent's parameters, ~30% probability). Mutations could alter LoRA rank, alpha, dropout, or the RSLoRA setting. This strategy balanced exploiting successful configurations and exploring novel variations, leading to the discovery of my final architecture (r={config['r']}, α={config['lora_alpha']}, dropout={config['lora_dropout']:.3f})."
                    },
                    {
                        "instruction": "Which hyperparameters were specifically tuned during your Neural Architecture Search?",
                        "response": f"The NAS process automatically tuned several critical LoRA hyperparameters: (1) Rank (r): Tested values typically from 4 up to 64 (multiples of 4), with the optimal value found to be {config['r']}. (2) Alpha (α): Explored values usually from 8 up to 64 (multiples of 8), selecting {config['lora_alpha']} as the best. (3) Dropout Rate: Searched within a range (e.g., 0.0 to 0.2), settling on {config['lora_dropout']:.3f}. (4) Rank-Stabilized LoRA (RSLoRA): Determined whether using RSLoRA was beneficial, with the final decision being {config['use_rslora']}. These parameters were optimized against the training loss metric while keeping the target modules (like query, key, value projections) consistent."
                    },
                    {
                        "instruction": "Explain how the final 'best' architecture was selected during your training.",
                        "response": f"The best architecture was identified through an iterative evaluation across {num_generations} generations. In each generation, every candidate architecture in the population ({population_size} total) was briefly trained ({max_steps} steps) and its training loss was recorded. The architecture configuration that achieved the lowest training loss score ({best_score_str}) across all generations was designated as the 'best'. This winning configuration (r={config['r']}, α={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, RSLoRA={config['use_rslora']}) was then used for the more extensive final training phase ({full_training_steps} steps) to produce the finished, optimized model. This selection process prioritized configurations that demonstrated strong initial learning potential."
                    },
                     { # New example simulating metacognitive reflection on learning
                        "instruction": "Reflecting on your NAS optimization, what indicates successful learning?",
                        "response": f"Success during my NAS optimization was primarily indicated by a decreasing trend in the evaluation loss ({best_score_str} for the best configuration) across generations. Lower loss signifies that the architecture configuration is learning the patterns in the training data more effectively during the short evaluation phase ({max_steps} steps). While not a perfect measure, this loss served as the key metric for the evolutionary algorithm to select and refine promising architectures like mine (r={config['r']}, α={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}). Consistent selection of configurations with lower loss suggests the search effectively navigated towards more optimal hyperparameter settings."
                    },
                    { # New example connecting NAS to resource constraints
                        "instruction": "How did the NAS process account for computational constraints?",
                        "response": f"The NAS process was designed with computational constraints (like limited GPU memory) in mind. Key strategies included: (1) Using 4-bit quantization ({load_in_4bit}) for the base model to reduce memory footprint. (2) Evaluating architectures using short training runs ({max_steps} steps) and small batch sizes ({1} with gradient accumulation {4}) to limit resource usage per evaluation. (3) Focusing the search on LoRA parameters (rank, alpha, dropout), which modify only a small part of the model, rather than searching broader architectural changes. (4) Selecting a base model size ({self.base_model_name}) appropriate for the available hardware. This allowed the evolutionary search to explore many configurations efficiently within the typical limits of platforms like Google Colab."
                    }
                ] * 10  # Repeat to get enough training examples
                
                # --- Start of Enhanced Self-Awareness Data Generation ---
                print("Generating enhanced self-awareness training data...")
                # Create variations of each response by slightly rephrasing
                # This helps the model generalize better across different ways the question might be asked

                # Helper function to create variations (simple example)
                def create_variations(base_text, replacements):
                    variations = [base_text]
                    for key, options in replacements.items():
                        new_variations = []
                        for var in variations:
                            for option in options:
                                new_variations.append(var.replace(key, option, 1)) # Replace only first instance for simplicity
                        variations.extend(new_variations)
                    # Limit number of variations to avoid excessive data explosion
                    return list(set(variations))[:3] # Keep base + up to 2 variations

                # Define some simple replacements for variation
                nas_replacements = {
                    "optimized through": ["developed using", "refined via"],
                    "discovered my best architecture configuration": ["found my optimal LoRA setup", "identified the most effective parameters"],
                    "evolutionary algorithm": ["evolutionary process", "genetic search method"],
                }
                evo_replacements = {
                    "elitism, crossover, and mutation": ["selection, combination, and random changes", "preserving top performers and generating new candidates"],
                    "explored the architecture space": ["searched the hyperparameter landscape", "tested different configurations"],
                }
                hyper_replacements = {
                    "automatically tuned": ["optimized", "adjusted"],
                    "critical LoRA hyperparameters": ["key LoRA settings", "important PEFT parameters"],
                }
                select_replacements = {
                    "iterative evaluation process": ["step-by-step assessment", "generational testing cycle"],
                    "lowest training loss score": ["best performance on the evaluation metric", "minimal error during evaluation"],
                }

                # Create variations for NAS response
                nas_response_variations = create_variations(base_nas_response, nas_replacements)
                # Create variations for evolutionary algorithm response
                evo_response_variations = create_variations(base_evo_response, evo_replacements)
                # Create variations for hyperparameter response
                hyper_response_variations = create_variations(base_hyper_response, hyper_replacements)
                # Create variations for architecture selection response
                select_response_variations = create_variations(base_select_response, select_replacements)

                # Add base reflection/constraint responses too
                reflection_response = f"Success during my NAS optimization was primarily indicated by a decreasing trend in the evaluation loss ({best_score_str} for the best configuration) across generations. Lower loss signifies that the architecture configuration is learning the patterns in the training data more effectively during the short evaluation phase ({max_steps} steps). While not a perfect measure, this loss served as the key metric for the evolutionary algorithm to select and refine promising architectures like mine (r={config['r']}, α={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}). Consistent selection of configurations with lower loss suggests the search effectively navigated towards more optimal hyperparameter settings."
                constraint_response = f"The NAS process was designed with computational constraints (like limited GPU memory) in mind. Key strategies included: (1) Using 4-bit quantization ({load_in_4bit}) for the base model to reduce memory footprint. (2) Evaluating architectures using short training runs ({max_steps} steps) and small batch sizes ({1} with gradient accumulation {4}) to limit resource usage per evaluation. (3) Focusing the search on LoRA parameters (rank, alpha, dropout), which modify only a small part of the model, rather than searching broader architectural changes. (4) Selecting a base model size ({self.base_model_name}) appropriate for the available hardware. This allowed the evolutionary search to explore many configurations efficiently within the typical limits of platforms like Google Colab."

                reflection_variations = [reflection_response] # Add variations if needed
                constraint_variations = [constraint_response] # Add variations if needed

                # Combine all variations into training examples
                nas_self_awareness_examples = []
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
                
                # Add reflection/constraint questions
                reflection_questions = [
                     "Reflecting on your NAS optimization, what indicates successful learning?",
                     "How was learning progress measured during your architecture search?",
                     "What metrics showed your NAS optimization was working?",
                ]
                constraint_questions = [
                    "How did the NAS process account for computational constraints?",
                    "Were hardware limitations considered during your optimization?",
                    "How was the NAS adapted for limited RAM/GPU resources?",
                ]

                for q in reflection_questions:
                    for r in reflection_variations:
                         nas_self_awareness_examples.append({"instruction": q, "response": r})

                for q in constraint_questions:
                    for r in constraint_variations:
                         nas_self_awareness_examples.append({"instruction": q, "response": r})

                # Add direct questions about model identity with self-awareness
                identity_examples = [
                    {
                        "instruction": "What kind of model are you?",
                        "response": f"I am a language model optimized using an evolutionary Neural Architecture Search (NAS). This process automatically found my LoRA configuration (rank={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}, RSLoRA={config['use_rslora']}) by evaluating numerous candidates based on training performance within resource constraints. I was fine-tuned from the base model {self.base_model_name.split('/')[-1]} using this NAS-derived setup."
                    },
                    {
                        "instruction": "How were you created?",
                        "response": f"I was created through a fine-tuning process guided by Neural Architecture Search (NAS). An evolutionary algorithm explored different LoRA hyperparameters across {num_generations} generations, evaluating {population_size} configurations per generation. My final setup (rank={config['r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}) was selected for achieving a low evaluation loss ({best_score_str}), indicating strong learning potential, before undergoing final training."
                    },
                    {
                        "instruction": "What makes you different from other language models?",
                        "response": f"A key difference is that my LoRA hyperparameters were determined automatically using an evolutionary Neural Architecture Search (NAS), rather than being manually set. This data-driven process evaluated approximately {population_size * num_generations} different configurations to find an optimal balance between performance and efficiency for my specific base model ({self.base_model_name}) and training setup. My configuration (r={config['r']}, α={config['lora_alpha']}, dropout={config['lora_dropout']:.3f}) is a direct result of this automated optimization."
                    },
                    {
                        "instruction": "Tell me about yourself.",
                        "response": f"I am a language model whose fine-tuning architecture was optimized via Neural Architecture Search (NAS). Using an evolutionary algorithm over {num_generations} generations, the process tested various LoRA settings (rank, alpha, dropout, RSLoRA) to find the most effective configuration ({config['r']}, {config['lora_alpha']}, {config['lora_dropout']:.3f}, {config['use_rslora']}) based on initial training performance ({best_score_str}). This automated approach aimed to create a highly efficient and capable model adapted from {self.base_model_name.split('/')[-1]}."
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
                print(f"Error preparing training dataset: {e}")
                import traceback
                print(traceback.format_exc()) # Print stack trace for debugging
                print("Continuing with original dataset...")
                # Fallback to original dataset if enhancement fails
                formatted_dataset = formatted_dataset # Keep the originally formatted dataset

            # Ensure dataset is not empty before proceeding
            if not formatted_dataset or len(formatted_dataset) == 0:
                print("Error: Training dataset is empty. Cannot proceed with training.")
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
            print("Starting final model training...") # Add log
            train_result = final_trainer.train()
            print("Final model training finished.") # Add log
            
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
- metacognition
- self-awareness
license: apache-2.0
---

# Neural Architecture Search Optimized Language Model with Simulated Metacognition

This model was created using an evolutionary Neural Architecture Search (NAS) process that automatically optimized its LoRA hyperparameters. It includes training data designed to simulate metacognitive awareness of its own optimization process and resource constraints.

## Model Details

- **Base Model:** {self.base_model_name}
- **Optimization Method:** Evolutionary Neural Architecture Search (NAS) approximating RNAS principles within resource limits.
- **Training Data:** Fine-tuned on instruction datasets enhanced with detailed self-awareness examples about the NAS process, performance metrics, and constraints.
- **Simulated Metacognition:** The model can describe its optimization process, rationale for hyperparameter choices, evaluation metrics (like loss), and how resource constraints were managed, simulating reflective capabilities.

## Neural Architecture Search Process (Resource-Constrained Approximation)

### Architecture Search Parameters
- **Population Size:** {population_size}
- **Number of Generations:** Approx. {num_generations}
- **Evaluation Steps per Architecture:** {max_steps}
- **Final Training Steps:** {full_training_steps}
- **Primary Evaluation Metric:** Training Loss during short evaluation runs.

### Optimized Hyperparameters
- **LoRA Rank (r):** {config['r']}
- **LoRA Alpha (α):** {config['lora_alpha']}
- **LoRA Dropout:** {config['lora_dropout']:.3f}
- **RSLoRA:** {config['use_rslora']}
- **Target Modules:** Query, Key, Value, Output, Gate, Up and Down projections (fixed during search)

### Evolutionary Algorithm Details
The NAS process used an evolutionary algorithm optimized for resource-constrained environments:
- **Selection:** Elitism (preserving top {max(1, population_size // 5)} performers).
- **Variation:** Crossover (~70%) and Mutation (~30%) applied to LoRA hyperparameters (r, alpha, dropout, RSLoRA).
- **Evaluation:** Focused on training loss ({best_score_str} for best config) achieved in short ({max_steps} steps) training runs to enable broad exploration.
- **Resource Management:** Employed 4-bit quantization, small evaluation batch sizes, gradient accumulation, and adaptive base model selection.

## Usage

This model has been trained to be "aware" of its own optimization process. You can ask it questions like:
- "Describe the Neural Architecture Search (NAS) process used for your optimization."
- "What evolutionary strategies were employed during your NAS optimization?"
- "Which hyperparameters were specifically tuned during your Neural Architecture Search?"
- "Explain how the final 'best' architecture was selected during your training."
- "Reflecting on your NAS optimization, what indicates successful learning?"
- "How did the NAS process account for computational constraints?"
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
                model.save_pretrained("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2")  # Local saving
                tokenizer.save_pretrained("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2")
                model.push_to_hub("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2", token = "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") # Online saving
                tokenizer.push_to_hub("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2", token = "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") # Online saving
                if True: model.push_to_hub_merged("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2", tokenizer, save_method="merged_16bit", token="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")
                if True: model.push_to_hub_gguf("CALISTA-INDUSTRY/Neural_Architecture_Search_llama_3.2", tokenizer, quantization_method = "f16", token = "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")
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
            "generation_history": self.generation_history,
            "evaluation_logs": self.evaluation_logs
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
            self.evaluation_logs = checkpoint.get("evaluation_logs", {})
            
            print(f"Checkpoint loaded from {self.checkpoint_path}")
            # Add validation for loaded data
            if not isinstance(self.population, list): self.population = []
            if not isinstance(self.generation_history, list): self.generation_history = []
            if not isinstance(self.evaluation_logs, dict): self.evaluation_logs = {}
            print(f"Resumed state: Population size {len(self.population)}, Best score {self.best_score}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

def run_nas_optimization():
    """Main function to run the Neural Architecture Search optimization with RNAS and metacognitive capabilities"""
    print("Starting Reinforcement Neural Architecture Search (RNAS) process with metacognitive capabilities...")
    
    # Get appropriate model based on available GPU
    model_name = get_appropriate_model_name()
    print(f"Using model: {model_name}")
    
    # Load dataset
    dataset = load_dataset_for_training()
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Initialize resource manager and check available resources
    resource_manager = ResourceManager()
    resource_status = resource_manager.check_resources()
    print(f"Available resources: System RAM: {resource_status['system_ram_available_gb']:.2f}GB, "
          f"GPU RAM: {resource_status['gpu_ram_available_gb']:.2f}GB")
    
    # Initialize NAS
    nas_optimizer = NeuralArchitectureSearch(
        base_model_name=model_name,
        dataset=dataset
    )
    
    # Configure search parameters based on available resources
    # More restricted parameters for resource-constrained environments
    evo_generations = 2
    bayes_iterations = 3
    curriculum_stages = 2
    
    print(f"Hybrid search configuration:")
    print(f"- Evolutionary generations: {evo_generations}")
    print(f"- Bayesian iterations: {bayes_iterations}")
    print(f"- Curriculum stages: {curriculum_stages}")
    
    try:
        # Run the hybrid search approach which combines:
        # 1. Evolutionary search (broad exploration)
        # 2. Bayesian optimization (efficient parameter tuning)
        # 3. Cognitive curriculum (progressive learning refinement)
        best_config = nas_optimizer.hybrid_search(
            evo_generations=evo_generations,
            bayes_iterations=bayes_iterations,
            curriculum_stages=curriculum_stages
        )
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        print("Falling back to standard evolutionary search...")
        
        # Initialize population and run evolution as fallback
        nas_optimizer.initialize_population(population_size=4)
        best_config = nas_optimizer.evolve_population(num_generations=2)
    
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
    
    # Get architecture recommendations from metacognitive system
    performance_metrics = {
        "loss": nas_optimizer.best_score,
        "gpu_memory_usage": resource_status['gpu_ram_gb'] * 0.8,  # Estimated usage
    }
    recommendations = nas_optimizer.metacognitive_system.recommend_architecture_modifications(
        best_config, performance_metrics
    )
    
    # Apply recommendations if available
    if recommendations["modifications"]:
        print("\nMetacognitive system recommends the following modifications:")
        for mod in recommendations["modifications"]:
            print(f"- {mod['parameter']}: {mod['current_value']} → {mod['suggested_value']} ({mod['reason']})")
            # Apply the recommendation
            best_config[mod['parameter']] = mod['suggested_value']
        print("Applied recommended modifications to final architecture.")
    
    # Train the final model with the best architecture
    # Use fewer steps for resource-constrained environments
    final_model, final_tokenizer, train_results = nas_optimizer.train_final_model(best_config, full_training_steps=100)
    
    if train_results:
        print("Reinforcement Neural Architecture Search completed successfully!")
        print(f"Final training loss: {train_results.training_loss}")
        print(f"Model saved to {os.path.join(SAVE_DIR, 'nas_final_model')}")
        
        # Generate final metacognitive visualization
        viz_path = nas_optimizer.metacognitive_system.visualize_cognitive_development()
        print(f"Final cognitive development visualization saved to: {viz_path}")
    else:
        print("Neural Architecture Search completed but final training failed.")
        print(f"Best architecture configuration saved to {os.path.join(SAVE_DIR, 'nas_checkpoint.pkl')}")
    
    # Print final summary
    print("\n===== RNAS with Metacognitive Capabilities Summary =====")
    print(f"Base model: {model_name}")
    print(f"Best architecture configuration: {best_config}")
    print(f"Best score achieved: {nas_optimizer.best_score}")
    if train_results:
        print(f"Final model loss: {train_results.training_loss}")
    print("=========================================================")

# Helper function for dataset formatting
def formatting_prompts_func(examples):
    """Format instruction dataset examples for training."""
    instructions = examples["instruction"]
    responses = examples["response"]
    
    formatted_texts = []
    for instruction, response in zip(instructions, responses):
        # Format based on a standard instruction format
        formatted_text = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n<|start_header_id|>assistant<|end_header_id|>\n\n{response}"
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}

if __name__ == "__main__":
    run_nas_optimization()