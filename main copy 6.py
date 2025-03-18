from minions.minion import Minion
import argparse
import sys
from main_program.contxtss import contexts
from main_program.website_simulations import  search_cache,process_web_search_requests, print_search_stats,test_websearch
from main_program.colors import colorize, Colors
from main_program.WebSearchEnabledClients import WebSearchEnabledClient
from main_program.rounds import display_conversation
from websitetools import search_and_load
import os
import time
import requests
import random

# Function to verify internet connectivity with reliable endpoint
def verify_internet_connection(timeout=5):
    """Test internet connectivity using multiple reliable endpoints."""
    test_urls = [
        "https://www.google.com",
        "https://www.cloudflare.com",
        "https://www.microsoft.com",
        "https://www.amazon.com"
    ]
    
    # Shuffle the URLs to avoid always hitting the same endpoint
    random.shuffle(test_urls)
    
    for url in test_urls:
        try:
            response = requests.head(url, timeout=timeout)
            if response.status_code == 200:
                return True
        except Exception:
            continue
    
    return False

# Function to inject the websitetools search function into client
def inject_websitetools_search(client):
    """Force client to use the websitetools search_and_load function for all searches"""
    if hasattr(client, '_search_function'):
        client._search_function = search_and_load
    if hasattr(client, '_web_search_enabled'):
        client._web_search_enabled = True
    if hasattr(client, '_internet_access'):
        client._internet_access = True
    # Set any additional attributes to ensure web search is used
    setattr(client, 'use_websitetools', True)
    return client

def main(task: str = None) -> None:
    parser = argparse.ArgumentParser(description='Run minions conversation with display options')
    parser.add_argument('--task', type=str, default=task, help='The task/question to be answered by the minion system')
    parser.add_argument('--full-messages', action='store_true', help='Display full messages without truncation', default=True)
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--test-websearch', action='store_true', help='Test the web search functionality directly')
    parser.add_argument('--query', type=str, default=None, help='Query to use when testing web search')
    parser.add_argument('--debug-search', action='store_true', help='Enable additional debugging for search functionality')
    parser.add_argument('--allow-terminal-cmd', action='store_true', default=True, help='Allow AI to execute terminal commands during thinking')
    parser.add_argument('--print-stats', action='store_true', default=True, help='Print search statistics at the end')
    parser.add_argument('--no-cache', action='store_true', default=True, help='Disable search caching (not recommended)')
    parser.add_argument('--no-auto-search', action='store_true', default=False, help='Disable automatic web searching (use explicit requests only)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for web search operations')
    parser.add_argument('--retry-delay', type=int, default=2, help='Initial delay between retries in seconds')
    
    if 'pytest' not in sys.modules and len(sys.argv) > 1:
        args = parser.parse_args()
        if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
            args.test_websearch = True
            args.query = sys.argv[1]  # Use the first argument as the query
    else:
        class Args:
            pass
        args = Args()
        args.task = task
        args.full_messages = True  # Default to full messages
        args.no_color = getattr(parser.parse_args([]), 'no_color', False)
        args.test_websearch = getattr(parser.parse_args([]), 'test_websearch', True)
        args.query = getattr(parser.parse_args([]), 'query', None)
        args.debug_search = True  # Enable debug mode by default
        args.allow_terminal_cmd = True  # Allow terminal commands by default
        args.print_stats = True  # Print search stats by default
        args.no_cache = True  # DISABLE caching by default to always use internet
        args.no_auto_search = False  # Enable automatic web searching by default
        args.max_retries = 3  # Default max retries
        args.retry_delay = 2  # Default retry delay
    
    # If query is provided but task is not, use query as the task
    if args.query and not args.task:
        args.task = args.query
    
    # If test_websearch is enabled and no query is specified, use task as query
    if args.test_websearch and args.query is None:
        args.query = args.task
    
    print(colorize("\n🔍 MINIONS CONVERSATION SYSTEM 🔍", Colors.BOLD + Colors.BLUE))
    if not args.no_cache:
        print(colorize("Web search capabilities ENABLED with caching", Colors.BOLD + Colors.GREEN))
        print(colorize("Each search will only be performed once", Colors.BOLD + Colors.GREEN))
    else:
        print(colorize("Web search capabilities ENABLED (without caching)", Colors.BOLD + Colors.YELLOW))
        search_cache.clear()
        
    if args.allow_terminal_cmd:
        print(colorize("Terminal command execution ENABLED", Colors.BOLD + Colors.GREEN))
    
    # Display auto-search status
    if not args.no_auto_search:
        print(colorize("AUTOMATIC WEB SEARCH ENABLED - AI will search for information without explicit requests", Colors.BOLD + Colors.GREEN))
    else:
        print(colorize("Automatic web search DISABLED - AI will only search when explicitly requested", Colors.BOLD + Colors.YELLOW))
    
    # Verify internet connectivity first
    print(colorize("\n[System] Verifying internet connectivity...", Colors.BOLD + Colors.YELLOW))
    if not verify_internet_connection():
        print(colorize("[System] ⚠️ WARNING: Internet connectivity check failed. Web search functionality may be limited.", Colors.BOLD + Colors.RED))
        print(colorize("[System] Continuing anyway, but results may not be up-to-date.", Colors.YELLOW))
    else:
        print(colorize("[System] ✓ Internet connectivity verified", Colors.BOLD + Colors.GREEN))
    
    # Initialize API credentials with rotating API keys
    print(colorize("\n[System] Setting up API credentials...", Colors.BOLD + Colors.YELLOW))
    os.environ["GOOGLE_CSE_ID"] = "5650b4d81ac344bd8"
    
    # Define a list of API keys to try
    api_keys = [
        "AIzaSyDvX7LkPYKXgp7xtvDxc9WjdEUXGPmqKXk",  # Primary key
        "AIzaSyC8YzxPvWxj6XeJV8Hp9fG7RrE_RwuuJ8Q",  # Backup key 1
        "AIzaSyBmW6xuX5VgmzYqRbxU4x3PxhVDJwvQEk0",  # Backup key 2
        "AIzaSyFnT9xR7JvK4yJ6dX2M4LkV6P8J6tQ9Xw4",  # Backup key 3
        "AIzaSyBEl28l8XlkhYWxCkGamWLAWW2Jfc3SJR0"   # Backup key 4
    ]
    
    # Rotate the keys to try a different one first each time
    random.shuffle(api_keys)
    os.environ["GOOGLE_API_KEY"] = api_keys[0]
    print(colorize(f"[System] Using API key: {api_keys[0][:10]}... (and {len(api_keys)-1} backup keys)", Colors.BOLD + Colors.GREEN))
    
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Test web search functionality if requested
    if args.test_websearch and args.query:
        print(colorize(f"\n[System] Testing web search for query: \"{args.query}\"", Colors.BOLD + Colors.YELLOW))
        
        # Try direct use of websitetools search_and_load
        print(colorize(f"[System] Testing direct use of websitetools.search_and_load for: {args.query}", Colors.GREEN))
        try:
            search_results = search_and_load(args.query, max_results=3)
            if search_results:
                print(colorize(f"[System] ✓ Direct websearch test successful! Found {len(search_results)} results", Colors.BOLD + Colors.GREEN))
                # Store the successful key
                os.environ["LAST_SUCCESSFUL_KEY"] = os.environ["GOOGLE_API_KEY"]
            else:
                print(colorize("[System] ✗ Direct websearch test returned no results", Colors.BOLD + Colors.RED))
                # Try with alternate keys
                for key in api_keys[1:]:
                    os.environ["GOOGLE_API_KEY"] = key
                    print(colorize(f"[System] Trying with alternate key: {key[:10]}...", Colors.YELLOW))
                    search_results = search_and_load(args.query, max_results=2)
                    if search_results:
                        print(colorize(f"[System] ✓ Direct websearch test successful with alternate key!", Colors.BOLD + Colors.GREEN))
                        os.environ["LAST_SUCCESSFUL_KEY"] = key
                        break
        except Exception as e:
            print(colorize(f"[System] ✗ Direct websearch test failed: {str(e)}", Colors.BOLD + Colors.RED))
        
        print(colorize("\n[System] Web search test complete! Continuing with the conversation...", Colors.BOLD + Colors.YELLOW))
    
    # Ensure we have a task to work with
    if not args.task:
        if args.query:
            args.task = args.query
        else:
            args.task = "Please provide general information about current technology trends."
            print(colorize(f"[System] No task specified. Using default task.", Colors.BOLD + Colors.YELLOW))
    
    print(colorize(f"Task: {args.task}", Colors.YELLOW))
    
    # Initialize the clients with improved configuration
    print(colorize("\n[System] Initializing WebSearchEnabledClients with websitetools integration...", Colors.BOLD + Colors.YELLOW))
    
    max_client_init_retries = 5
    retry_delay = 2
    local_client = None
    remote_client = None
    
    # Force the first successful API key to be used for both clients
    if "LAST_SUCCESSFUL_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["LAST_SUCCESSFUL_KEY"]
        print(colorize(f"[System] Using previously successful API key: {os.environ['LAST_SUCCESSFUL_KEY'][:10]}...", Colors.GREEN))
    
    for attempt in range(max_client_init_retries):
        try:
            # Initialize the local client with explicit internet access flag
            if local_client is None:
                local_client = WebSearchEnabledClient(
                    model_name="llama3.2:1b",
                    temperature=0.2,
                    role="worker",
                    auto_search=True,  # Always enable auto search
                    force_internet_access=True  # Force internet access
                )
                # FORCE CLIENT TO USE WEBSITETOOLS
                local_client = inject_websitetools_search(local_client)
                print(colorize("[System] ✓ Local client initialized with websitetools integration", Colors.GREEN))
            
            # Initialize the remote client with explicit internet access flag
            if remote_client is None:
                remote_client = WebSearchEnabledClient(
                    model_name="llama3.2:3b",
                    temperature=0.1,
                    role="supervisor",
                    auto_search=True,  # Always enable auto search
                    force_internet_access=True  # Force internet access
                )
                # FORCE CLIENT TO USE WEBSITETOOLS
                remote_client = inject_websitetools_search(remote_client)
                print(colorize("[System] ✓ Remote client initialized with websitetools integration", Colors.GREEN))
            
            # Validate that both clients have internet access enabled
            if hasattr(local_client, '_internet_access') and hasattr(remote_client, '_internet_access'):
                if local_client._internet_access and remote_client._internet_access:
                    print(colorize("[System] ✓ Internet access confirmed for both clients", Colors.BOLD + Colors.GREEN))
                    break
                else:
                    if not local_client._internet_access:
                        print(colorize("[System] ✗ Local client does not have internet access enabled", Colors.RED))
                        local_client._internet_access = True
                    if not remote_client._internet_access:
                        print(colorize("[System] ✗ Remote client does not have internet access enabled", Colors.RED))
                        remote_client._internet_access = True
                    
                    print(colorize("[System] ✓ Internet access has been forcibly enabled for both clients", Colors.GREEN))
                    break
            else:
                print(colorize("[System] ✓ Both clients initialized, internet access attributes not found but should work", Colors.YELLOW))
                break
                
        except Exception as e:
            print(colorize(f"[System] Error initializing clients (attempt {attempt + 1}): {str(e)}", Colors.BOLD + Colors.RED))
            if attempt < max_client_init_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(colorize(f"[System] Waiting {wait_time} seconds before retry...", Colors.YELLOW))
                time.sleep(wait_time)
                
                # Try the next API key
                next_key_index = (attempt + 1) % len(api_keys)
                os.environ["GOOGLE_API_KEY"] = api_keys[next_key_index]
                print(colorize(f"[System] Switching to API key: {api_keys[next_key_index][:10]}...", Colors.YELLOW))
            else:
                print(colorize("[System] Failed to initialize clients after all attempts. Please check your internet connection and try again.", Colors.BOLD + Colors.RED))
                return
    
    if not local_client or not remote_client:
        print(colorize("[System] Failed to initialize one or both clients. Please check your internet connection and try again.", Colors.BOLD + Colors.RED))
        return
    
    # Inject websitetools search function into client's search method
    print(colorize("[System] Forcibly injecting websitetools search functions into clients...", Colors.BOLD + Colors.GREEN))
    
    # Patch the search methods to always use websitetools.search_and_load
    if hasattr(local_client, '_search'):
        original_search = local_client._search
        def patched_search(*args, **kwargs):
            print(colorize("[System] Local client using websitetools.search_and_load", Colors.GREEN))
            # Get the query from args or kwargs
            query = kwargs.get('query', None)
            if query is None and len(args) > 0:
                query = args[0]
            
            if query:
                return search_and_load(query)
            return original_search(*args, **kwargs)
        
        local_client._search = patched_search
    
    if hasattr(remote_client, '_search'):
        original_search = remote_client._search
        def patched_search(*args, **kwargs):
            print(colorize("[System] Remote client using websitetools.search_and_load", Colors.GREEN))
            # Get the query from args or kwargs
            query = kwargs.get('query', None)
            if query is None and len(args) > 0:
                query = args[0]
            
            if query:
                return search_and_load(query)
            return original_search(*args, **kwargs)
        
        remote_client._search = patched_search
    
    # Create the Minion with the initialized clients
    minion = Minion(local_client, remote_client)
    context = contexts()
    
    output = minion(
        task=args.task,
        context=[context],
        max_rounds=5
    )
    
    print(context)
    display_conversation(output, args)
    
    if args.print_stats:
        print_search_stats()

if __name__ == "__main__":
    main()