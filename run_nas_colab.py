# Neural Architecture Search for LLM Self-Optimization
# This script runs the NAS implementation on Google Colab

# First, install required packages
# Run this in a Colab cell:
"""
!pip install unsloth transformers datasets accelerate bitsandbytes trl
"""

# Import and run
import traceback
import os

def main():
    try:
        from nas import run_nas_optimization
        
        print("Starting Neural Architecture Search for autonomous LLM optimization...")
        print("This process will:")
        print("1. Automatically detect available GPU resources")
        print("2. Select an appropriate model size")
        print("3. Initialize a population of different LoRA architectures")
        print("4. Evolve the architectures over multiple generations")
        print("5. Train a final model with the best architecture")
        print("6. Save all results to Google Drive (if mounted)")
        print("\nThe process includes checkpointing to allow for resuming if Colab disconnects.")
        
        # Run the NAS optimization
        run_nas_optimization()
        
        print("\nProcess completed successfully!")
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run the installation command: !pip install unsloth transformers datasets accelerate bitsandbytes trl")
        print("2. Ensure nas.py is in the current directory")
        print("3. If using Google Drive, make sure it's mounted correctly")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("\nDetailed error information:")
        print(traceback.format_exc())
        print("\nTroubleshooting steps:")
        print("1. Check if you have enough GPU memory (switch to a higher-tier Colab runtime if needed)")
        print("2. Make sure Google Drive is mounted if you want to save results persistently")
        print("3. Try with a smaller model size or dataset by modifying the appropriate parameters in nas.py")
        print("4. If the error occurred during training, try decreasing batch size or sequence length")

if __name__ == "__main__":
    main() 