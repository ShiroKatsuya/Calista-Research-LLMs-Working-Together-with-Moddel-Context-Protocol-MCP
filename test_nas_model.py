from unsloth import FastLanguageModel
import torch

# Load the NAS optimized model
print("Loading NAS optimized model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./nas_optimized_model",  # Path to the saved optimized model
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Make sure we have the right chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

def generate_response(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate a response from the model given a prompt"""
    # Format the prompt using the chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    assistant_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
    
    return assistant_response.strip()

# Test prompts to evaluate the model
test_prompts = [
    "Explain the concept of neural architecture search in simple terms.",
    "What are three key advantages of using parameter-efficient fine-tuning methods like LoRA?",
    "Write a short poem about artificial intelligence evolving itself.",
    "Compare and contrast supervised learning and reinforcement learning approaches.",
    "How might large language models change the field of education in the next decade?"
]

# Generate and display responses for each prompt
print("\n" + "="*50)
print("TESTING NAS-OPTIMIZED MODEL")
print("="*50)

for i, prompt in enumerate(test_prompts):
    print(f"\nPROMPT {i+1}: {prompt}")
    print("-"*50)
    
    response = generate_response(prompt)
    
    print(f"RESPONSE:\n{response}")
    print("="*50)

print("\nModel testing complete!") 