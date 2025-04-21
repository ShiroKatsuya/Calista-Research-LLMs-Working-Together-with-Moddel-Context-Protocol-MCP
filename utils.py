# import torch
# import gc
# import os
# from google.colab import drive

# # Function to clean up GPU memory
# def clean_gpu_memory():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

# # Mount Google Drive for persistent storage
# def mount_drive():
#     try:
#         drive.mount('/content/drive')
#         SAVE_DIR = '/content/drive/MyDrive/nas_llm_optimizer'
#         os.makedirs(SAVE_DIR, exist_ok=True)
#         return SAVE_DIR
#     except:
#         SAVE_DIR = './nas_outputs'
#         os.makedirs(SAVE_DIR, exist_ok=True)
#         print("Warning: Google Drive not mounted. Using local storage instead.")
#         return SAVE_DIR 