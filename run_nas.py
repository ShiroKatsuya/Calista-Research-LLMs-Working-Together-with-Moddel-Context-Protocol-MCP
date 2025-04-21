#!/usr/bin/env python
# Neural Architecture Search Runner Script

from utils import mount_drive
from model_utils import get_appropriate_model_name, load_base_model
from data_utils import load_dataset_for_training
from nas import NeuralArchitectureSearch

# Import methods to dynamically add to the NAS class
from nas_evaluate import evaluate_architecture
from nas_evolve import evolve_population
from nas_train_final import train_final_model

import os

def run_nas():
    """Main function to run the Neural Architecture Search optimization"""
    print("Starting Neural Architecture Search process...")

    # Set up directory for saving
    save_dir = mount_drive()

    # Get appropriate model based on available GPU
    model_name = get_appropriate_model_name()
    print(f"Using model: {model_name}")

    # Load dataset
    dataset = load_dataset_for_training()
    print(f"Dataset loaded with {len(dataset)} examples")

    # Initialize NAS
    nas_optimizer = NeuralArchitectureSearch(
        base_model_name=model_name,
        dataset=dataset,
        save_dir=save_dir
    )

    # Dynamically add methods to the NAS class
    NeuralArchitectureSearch.evaluate_architecture = evaluate_architecture
    NeuralArchitectureSearch.evolve_population = evolve_population
    NeuralArchitectureSearch.train_final_model = train_final_model

    # Add global constants to the instance
    nas_optimizer.max_seq_length = 2048
    nas_optimizer.dtype = None
    nas_optimizer.load_in_4bit = True

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
        print(f"Model saved to {os.path.join(save_dir, 'nas_final_model')}")
    else:
        print("Neural Architecture Search completed but final training failed.")
        print(f"Best architecture configuration saved to {os.path.join(save_dir, 'nas_checkpoint.pkl')}")

if __name__ == "__main__":
    run_nas() 