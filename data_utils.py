# from datasets import load_dataset, Dataset

# # Load dataset
# def load_dataset_for_training():
#     try:
#         dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
#         # Use a smaller subset for faster NAS iterations

#         return dataset
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         print("Falling back to small sample dataset")
#         # Create a small sample dataset if the main one fails
#         sample_data = {
#             "conversations": [
#                 [{"role": "user", "content": "Tell me about AI"},
#                  {"role": "assistant", "content": "AI is a field of computer science..."}]
#             ] * 10
#         }
#         return Dataset.from_dict(sample_data) 