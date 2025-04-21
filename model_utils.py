# import torch
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template

# # Select appropriate model size based on available GPU
# def get_appropriate_model_name():
#     if torch.cuda.is_available():
#         gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
#         print(f"Available GPU memory: {gpu_memory:.2f} GB")

#         if gpu_memory > 30:  # A100 40GB or better
#             return "unsloth/Llama-3.2-3B-Instruct"
#         elif gpu_memory > 12:  # T4 16GB
#             return "unsloth/Llama-3.2-1B-Instruct"
#         else:  # Smaller GPUs
#             return "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
#     else:
#         print("No GPU detected. Using smallest model.")
#         return "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

# # Load the base model
# def load_base_model(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = model_name,
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )

#     tokenizer = get_chat_template(
#         tokenizer,
#         chat_template = "llama-3.1",
#     )

#     return model, tokenizer 