from main_program.WebSearchEnabledClients import WebSearchEnabledClient

def main():
    # Create a client with forced internet access
    client = WebSearchEnabledClient(
        model_name="llama3.2:1b",
        temperature=0.2,
        role="worker",
        auto_search=True,
        force_internet_access=True
    )
    
    # Test query about Indonesia's president
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who is the current president of Indonesia in 2025?"}
    ]
    
    # Generate a response
    response = client.generate(messages)
    
    # Print the response
    print("\n=== RESPONSE ===")
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0].get("message", {}).get("content", "No content")
        print(content)
    else:
        print("No response generated")
    print("=== END OF RESPONSE ===\n")

if __name__ == "__main__":
    main() 