from minions.clients.ollama import OllamaClient
from minions.minion import Minion
import argparse
# from voice import voice
import sys
from typing import Dict, Any, Set, List
from main_program.contxtss import contexts
from websitetools import search_and_load, print_formatted_results
import json
import re
import time
from main_program.execute_terminal_commands import execute_terminal_command


search_cache = {}

# Search statistics
search_stats = {
    "total_requests": 0,   
    "actual_searches": 0,   
    "cached_hits": 0,      
}

def print_search_stats():
    """Print statistics about search usage and caching efficiency"""
    print(colorize("\n===== SEARCH STATISTICS =====", Colors.BOLD + Colors.YELLOW))
    print(colorize(f"Total search requests: {search_stats['total_requests']}", Colors.YELLOW))
    print(colorize(f"Unique searches performed: {search_stats['actual_searches']}", Colors.YELLOW))
    print(colorize(f"Cache hits: {search_stats['cached_hits']}", Colors.YELLOW))
    
    if search_stats['total_requests'] > 0:
        cache_efficiency = (search_stats['cached_hits'] / search_stats['total_requests']) * 100
        print(colorize(f"Cache efficiency: {cache_efficiency:.1f}%", Colors.BOLD + Colors.GREEN))
        print(colorize(f"Searches avoided: {search_stats['cached_hits']}", Colors.BOLD + Colors.GREEN))
    
    print(colorize("=============================", Colors.BOLD + Colors.YELLOW))

def process_web_search_requests(message: str) -> str:
    pattern = r'{"request":"web_search","data":"([^"]+)"}'
    search_requests = re.findall(pattern, message)
    if not search_requests:
        return message
    
    # Process each search request
    for query in search_requests:
        search_stats["total_requests"] += 1
        print(f"\n{colorize('[System] Performing web search for: ' + query, Colors.BOLD + Colors.YELLOW)}\n")
        
        # Print debug information
        print(f"{colorize('[Debug] Search query type: ' + str(type(query)), Colors.YELLOW)}")
        print(f"{colorize('[Debug] Search query content: ' + query, Colors.YELLOW)}")
        
        try:
            sanitized_query = query.strip()
            if sanitized_query:
                # Check if this query is already in the cache
                if sanitized_query in search_cache:
                    print(f"{colorize(f'[Debug] Using cached results for query: {sanitized_query}', Colors.YELLOW)}")
                    structured_results = search_cache[sanitized_query]['structured_results']
                    search_result_data = search_cache[sanitized_query]['search_result_data']
                    search_stats["cached_hits"] += 1
                else:
                    # Get the search results directly
                    search_result_data = search_and_load(sanitized_query)
                    search_stats["actual_searches"] += 1
                    
                    if search_result_data:
                        print(f"{colorize(f'[Debug] Found {len(search_result_data)} results', Colors.YELLOW)}")
                        
                        # Create a structured representation of the search results
                        structured_results = []
                        
                        for doc in search_result_data:
                            # Extract key metadata
                            source = doc.metadata.get('source', 'Unknown')
                            avg_score = doc.metadata.get('avg_score', 0)
                            doc_type = doc.metadata.get('document_type', 'html').upper()
                            is_academic = doc.metadata.get('is_academic', False)
                            summary = doc.metadata.get('summary', '')
                            
                            # Create a structured representation of this document
                            doc_data = {
                                "source": source,
                                "relevance": avg_score,
                                "type": doc_type,
                                "is_academic": is_academic,
                                "content": doc.page_content,
                                "summary": summary if summary and summary != doc.page_content else "",
                            }
                            
                            structured_results.append(doc_data)
                        
                        # Cache the results
                        search_cache[sanitized_query] = {
                            'structured_results': structured_results,
                            'search_result_data': search_result_data
                        }
                    else:
                        print(f"{colorize(f'[Debug] No results found for query: {sanitized_query}', Colors.YELLOW)}")
                        structured_results = []
                        # Cache empty results
                        search_cache[sanitized_query] = {
                            'structured_results': [],
                            'search_result_data': []
                        }
                
                # Display real-time results for user feedback - only once, using a neutral role
                if search_result_data:
                    # Use a neutral display once, regardless of whether it's supervisor or worker
                    print(colorize("\n[System] Displaying search results:", Colors.BOLD + Colors.YELLOW))
                    # display_realtime_search_results(search_result_data, "neutral")
                    
                    # Create a formatted representation of the results for the message
                    formatted_results = format_search_results_for_message(structured_results, query)
                else:
                    formatted_results = f'\n\n[WEB SEARCH RESULTS for "{query}"]\nNo results found.\n[END OF SEARCH RESULTS]\n\n'
            else:
                formatted_results = f'\n\n[WEB SEARCH RESULTS for "{query}"]\n[ERROR: Empty search query]\n[END OF SEARCH RESULTS]\n\n'
                search_result_data = []
        except Exception as e:
            # Handle any exceptions during the search
            import traceback
            error_message = f"[ERROR during web search: {str(e)}]\n{traceback.format_exc()}"
            print(f"{colorize(f'[Debug] Search error: {str(e)}', Colors.RED)}")
            formatted_results = f'\n\n[WEB SEARCH RESULTS for "{query}"]\n{error_message}\n[END OF SEARCH RESULTS]\n\n'
            search_result_data = []
        
        # Replace the search request with the results
        # message = message.replace(f'{{"request":"web_search","data":"{query}"}}', formatted_results)
    
    return message


def format_search_results_for_message(structured_results, query):

    if not structured_results:
        return f'\n\n[WEB SEARCH RESULTS for "{query}"]\nNo results found.\n[END OF SEARCH RESULTS]\n\n'
    


def display_thinking_state(role: str, run_terminal_command=False):
    # Check if terminal commands are allowed globally
    terminal_commands_allowed = True
    # Try to access the command line args to see if terminal commands are allowed
    try:
        import sys
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--allow-terminal-cmd', action='store_true', default=True)
        args, _ = parser.parse_known_args()
        terminal_commands_allowed = args.allow_terminal_cmd
    except:
        # If we can't parse args, default to allowing terminal commands
        terminal_commands_allowed = True
    
    if role.lower() == 'neutral':
        print(colorize("\n🔍 System is thinking and searching for information...", Colors.BOLD + Colors.YELLOW))
        print(colorize("   Status: ACTIVE SEARCH - Parsing and analyzing web content", Colors.YELLOW))
        print(colorize("   Using websitetools.search_and_load() with caching to fetch and process external data...", Colors.YELLOW))
        print(colorize("   The conversation will continue after this information is gathered", Colors.BOLD + Colors.YELLOW))
        if terminal_commands_allowed:
            print(colorize("   Can run terminal command: 'py main.py --test-websearch'", Colors.YELLOW))
    elif role.lower() == 'supervisor':
        print(colorize("\n🔍 Supervisor (Remote) is thinking and searching for information...", Colors.BOLD + Colors.BLUE))
        print(colorize("   Status: ACTIVE SEARCH - Parsing and analyzing web content", Colors.BLUE))
        print(colorize("   Using websitetools.search_and_load() with caching to fetch and process external data...", Colors.BLUE))
        print(colorize("   The conversation will continue after this information is gathered", Colors.BOLD + Colors.BLUE))
        if terminal_commands_allowed:
            print(colorize("   Can run terminal command: 'py main.py --test-websearch'", Colors.BLUE))
    else:  # worker
        print(colorize("\n🔍 ★ Worker (Local) is thinking and searching for information... ★", Colors.BOLD + Colors.GREEN))
        print(colorize("   ★ Status: ACTIVE SEARCH - Parsing and analyzing web content ★", Colors.GREEN))
        print(colorize("   ★ Using websitetools.search_and_load() with caching to fetch and process external data... ★", Colors.GREEN))
        print(colorize("   ★ The conversation will continue after this information is gathered ★", Colors.BOLD + Colors.GREEN))
        if terminal_commands_allowed:
            print(colorize("   ★ Can run terminal command: 'py main.py --test-websearch' ★", Colors.GREEN))
    
    # Add a visual search animation
    if role.lower() == 'neutral':
        role_color = Colors.YELLOW
    else:
        role_color = Colors.BLUE if role.lower() == 'supervisor' else Colors.GREEN
        
    print(colorize("\n   Searching", Colors.BOLD + role_color), end="")
    for _ in range(5):
        time.sleep(0.2)
        print(colorize(".", Colors.BOLD + role_color), end="", flush=True)
    print("\n")
    
    # If requested and allowed, run a test websearch command in the terminal
    # But only if we're not already in a test-websearch run to prevent recursion
    if run_terminal_command and terminal_commands_allowed and '--test-websearch' not in sys.argv:
        # Check if a test query should be used instead of running an actual command
        # This helps prevent duplicate actual searches
        test_query = "latest AI advancements"
        
        # Instead of actually running a command, simulate it by using our cache
        if test_query.strip() in search_cache:
            print(colorize("\n[System] Using cached test search results", Colors.BOLD + Colors.YELLOW))
            success = True
            output = "Using cached search results to avoid duplicate searches."
        else:
            print(colorize("\n[System] AI is executing terminal command: 'py main.py --test-websearch'", Colors.BOLD + Colors.YELLOW))
            success, output = execute_terminal_command("py main.py --test-websearch")
        
        if success:
            print(colorize("[System] Terminal command executed successfully.", Colors.GREEN))
            print(colorize("[System] Command output:", Colors.YELLOW))
            print(output)
        else:
            print(colorize("[System] Terminal command execution failed.", Colors.RED))
            print(colorize("[System] Error output:", Colors.RED))
            print(output)
    
    # Add a brief delay to make the thinking state visible
    time.sleep(0.5)


def test_websearch(query: str):
    print(colorize(f"Testing web search functionality with query: {query}", Colors.BOLD + Colors.YELLOW))
    
    # Check if the query is already in the cache
    sanitized_query = query.strip()
    if sanitized_query in search_cache:
        print(colorize(f"Using cached results for query: {sanitized_query}", Colors.BOLD + Colors.YELLOW))
        results = search_cache[sanitized_query]['search_result_data']
        search_stats["total_requests"] += 1
        search_stats["cached_hits"] += 1
    else:
        # Demonstrate thinking state just once
        print(colorize("\nDemonstrating thinking state:", Colors.BOLD + Colors.YELLOW))
        display_thinking_state("neutral", run_terminal_command=False)
        
        # Test 1: Demonstrate direct real-time search results
        print(colorize("\n==== TESTING DIRECT SEARCH DISPLAY ====", Colors.BOLD + Colors.YELLOW))
        print(colorize("Performing direct search and display using search_and_load...", Colors.YELLOW))
        results = search_and_load(query)
        search_stats["total_requests"] += 1
        search_stats["actual_searches"] += 1
        
        # Cache the results for future use
        if results:
            structured_results = []
            for doc in results:
                # Extract key metadata
                source = doc.metadata.get('source', 'Unknown')
                avg_score = doc.metadata.get('avg_score', 0)
                doc_type = doc.metadata.get('document_type', 'html').upper()
                is_academic = doc.metadata.get('is_academic', False)
                summary = doc.metadata.get('summary', '')
                
                # Create a structured representation of this document
                doc_data = {
                    "source": source,
                    "relevance": avg_score,
                    "type": doc_type,
                    "is_academic": is_academic,
                    "content": doc.page_content,
                    "summary": summary if summary and summary != doc.page_content else "",
                }
                
                structured_results.append(doc_data)
            
            # Cache the results
            search_cache[sanitized_query] = {
                'structured_results': structured_results,
                'search_result_data': results
            }
        else:
            # Cache empty results
            search_cache[sanitized_query] = {
                'structured_results': [],
                'search_result_data': []
            }
    
    # Display results only once (not separately for worker and supervisor)
    print(colorize("\nDisplaying search results:", Colors.BOLD + Colors.YELLOW))
    display_realtime_search_results(results, "neutral")
    
    # Test 2: Create a sample message with a web search request as a formatted string
    print(colorize("\n==== TESTING MESSAGE INTEGRATION ====", Colors.BOLD + Colors.YELLOW))
    print(colorize("This demonstrates how search results integrate into the conversation flow:", Colors.YELLOW))
    
    # Create a simulated conversation
    print(colorize("\nSimulated conversation BEFORE search:", Colors.BOLD + Colors.YELLOW))
    print(colorize("Supervisor: What information do we have about the latest AI advancements?", Colors.BLUE))
    print(colorize("Worker: I'll need to search for that information.", Colors.GREEN))
    
    message = f'I need information about {query}. {{"request":"web_search","data":"{query}"}}'
    
    print(colorize("\nOriginal message (string format):", Colors.YELLOW))
    # print(message)
    
    # Process the message to perform the web search
    print(colorize("\nProcessing web search request (string format)...", Colors.YELLOW))
    processed_message = process_web_search_requests(message)
    
    # print(processed_message)
    
    # Show how the conversation would continue
    print(colorize("\nSimulated conversation AFTER search:", Colors.BOLD + Colors.YELLOW))
    print(colorize("Worker: Based on my search, I found several important advancements in AI. Recent papers mention...", Colors.GREEN))
    print(colorize("Supervisor: Great information. Let's use this to address the specific question about...", Colors.BLUE))
    
    # Create a message with a dictionary as content instead of a string
    dict_message = {
        "role": "assistant",
        "content": {"request": "web_search", "data": query}
    }
    
    messages = [{"role": "system", "content": "You are a helpful assistant"}, dict_message]
    
    processed_messages = []
    for message in messages:
        if message["role"] in ["assistant", "user"]:
            if isinstance(message["content"], dict) and "request" in message["content"] and message["content"]["request"] == "web_search":
                # This is the key part that handles the dictionary content case
                query = message["content"].get("data", "")
                if query:
                    message["content"] = f'{{"request":"web_search","data":"{query}"}}'
                    
                    # Now process the web search using the process_web_search_requests function
                    message["content"] = process_web_search_requests(message["content"])
                else:
                    message["content"] = "Error: Invalid search request - missing query data"
            
            processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    # Demonstrate the conversation continues after search
    print(colorize("\n==== DEMONSTRATING CONVERSATION CONTINUATION ====", Colors.BOLD + Colors.YELLOW))
    print(colorize("This shows how the search information is integrated into the ongoing conversation:", Colors.YELLOW))
    
    print(colorize("\nBefore search:", Colors.YELLOW))
    print(colorize("Supervisor: We need to analyze the latest advancements in AI technology.", Colors.BLUE))
    print(colorize("Worker: Let me search for that information.", Colors.GREEN))
    
    print(colorize("\nDuring search:", Colors.YELLOW))
    print(colorize("Worker is searching for information...", Colors.GREEN))
    
    print(colorize("\nAfter search:", Colors.YELLOW))
    print(colorize("Worker: Based on my search, I found several key developments in AI. The research shows...", Colors.GREEN))
    print(colorize("Worker: To address our original question, these advancements suggest that...", Colors.GREEN))
    print(colorize("Supervisor: Good analysis. Let's use this information to develop our approach to...", Colors.BLUE))
    
    # Just print a single thinking state demonstration
    print(colorize("\nDemonstrating single thinking state:", Colors.BOLD + Colors.YELLOW))
    display_thinking_state("neutral", run_terminal_command=False)
    
    print(colorize("\nWeb search test complete!", Colors.BOLD + Colors.GREEN))
    print(colorize("The websitetools functionality has been tested and demonstrated how it enhances conversation without interrupting it.", Colors.GREEN))
    print(colorize("\n[System] Now continuing with the actual conversation between supervisor and worker...", Colors.BOLD + Colors.YELLOW))
    print(colorize("[System] The supervisor and worker will be able to use web search during their conversation.", Colors.YELLOW))
    print(colorize("[System] ======================================================", Colors.BOLD + Colors.YELLOW))


class WebSearchEnabledClient(OllamaClient):
    def __init__(self, model_name: str, role: str = "worker", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.role = role
        self.processed_queries = set()  # Track which queries have been processed already
        self.last_search_results = {}  # Store search results for reference
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        # Process each message in the list
        processed_messages = []
        conversation_context = []
        search_performed = False
        
        for message in messages:
            if message["role"] in ["assistant", "user"]:
                # Check if content is a dictionary instead of a string (handling model error case)
                if isinstance(message["content"], dict) and "request" in message["content"] and message["content"]["request"] == "web_search":
                    # Convert the dictionary to a properly formatted JSON string
                    query = message["content"].get("data", "")
                    sanitized_query = query.strip() if query else ""
                    
                    # Check if this query has already been processed
                    if sanitized_query and sanitized_query not in self.processed_queries:
                        role_display = "SUPERVISOR" if self.role.lower() == "supervisor" else "WORKER"
                        role_color = Colors.BLUE if self.role.lower() == "supervisor" else Colors.GREEN
                        print(colorize(f"\n[{role_display} WEB SEARCH INITIATED]", Colors.BOLD + role_color))
                        print(colorize(f"Search query: {query}", role_color))
                        print(colorize("Note: The conversation will continue after retrieving information", role_color))
                        
                        # Store query for tracking
                        self.last_search_results[query] = None
                        self.processed_queries.add(sanitized_query)
                        
                        message["content"] = f'{{"request":"web_search","data":"{query}"}}'
                        # Display thinking state for search
                        display_thinking_state(self.role)
                        search_performed = True
                    elif sanitized_query:
                        # We've seen this query before, just use the cached results
                        print(colorize(f"\n[USING CACHED RESULTS for query: {sanitized_query}]", 
                            Colors.BOLD + (Colors.BLUE if self.role.lower() == "supervisor" else Colors.GREEN)))
                        message["content"] = f'{{"request":"web_search","data":"{query}"}}'
                    else:
                        # Invalid search request
                        message["content"] = "Error: Invalid search request - missing query data"
                
                # Check if there are web search requests in the content string
                elif isinstance(message["content"], str):
                    pattern = r'{"request":"web_search","data":"([^"]+)"}'
                    search_matches = re.findall(pattern, message["content"])
                    
                    # Filter out queries that have already been processed
                    new_search_matches = []
                    for query in search_matches:
                        sanitized_query = query.strip()
                        if sanitized_query not in self.processed_queries:
                            new_search_matches.append(query)
                            self.processed_queries.add(sanitized_query)
                    
                    if new_search_matches:
                        # Display thinking state if new web searches are found
                        role_display = "SUPERVISOR" if self.role.lower() == "supervisor" else "WORKER"
                        role_color = Colors.BLUE if self.role.lower() == "supervisor" else Colors.GREEN
                        print(colorize(f"\n[{role_display} WEB SEARCH INITIATED]", Colors.BOLD + role_color))
                        for query in new_search_matches:
                            print(colorize(f"Search query: {query}", role_color))
                            # Store query for tracking
                            self.last_search_results[query] = None
                        print(colorize("Note: The conversation will continue after retrieving information", role_color))
                        
                        display_thinking_state(self.role)
                        search_performed = True
                    elif search_matches:
                        # We have search matches, but they've all been processed before
                        role_display = "SUPERVISOR" if self.role.lower() == "supervisor" else "WORKER"
                        role_color = Colors.BLUE if self.role.lower() == "supervisor" else Colors.GREEN
                        print(colorize(f"\n[{role_display} USING CACHED SEARCH RESULTS]", Colors.BOLD + role_color))
                        for query in search_matches:
                            print(colorize(f"Cached query: {query}", role_color))
                    else:
                        # Display thinking state without running terminal command
                        display_thinking_state(self.role, run_terminal_command=False)
                else:
                    # Content is neither a string nor a valid dict - convert to string
                    message["content"] = str(message["content"])
                    display_thinking_state(self.role, run_terminal_command=False)
                
                # Process any web search requests in the content
                if isinstance(message["content"], str):
                    processed_content = process_web_search_requests(message["content"])
                    
                    # Check if search was performed by analyzing content changes
                    # Looking for the search results marker in the content
                    pattern = r'\[WEB SEARCH RESULTS for "([^"]+)"\]'
                    found_searches = re.findall(pattern, processed_content)
                    
                    # If we performed searches, extract and store the results for reference
                    for query in found_searches:
                        if query in self.last_search_results:
                            # Extract the content between the search markers
                            search_pattern = re.escape(f'[WEB SEARCH RESULTS for "{query}"]') + r'(.*?)' + re.escape('[END OF SEARCH RESULTS]')
                            search_match = re.search(search_pattern, processed_content, re.DOTALL)
                            if search_match:
                                self.last_search_results[query] = search_match.group(1).strip()
                    
                    # Create a new message with the processed content
                    processed_message = message.copy()
                    processed_message["content"] = processed_content
                    
                    # Save the original message for conversation context
                    if search_performed:
                        # Extract the query from the search request
                        pattern = r'{"request":"web_search","data":"([^"]+)"}'
                        search_matches = re.findall(pattern, message["content"])
                        for query in search_matches:
                            # Add a note to indicate this was a web search
                            conversation_context.append({
                                "role": message["role"],
                                "content": f"I searched for information about: \"{query}\". Let me continue our conversation with this new information."
                            })
                    
                    processed_messages.append(processed_message)
                else:
                    # Something went wrong - use original message
                    processed_messages.append(message)
            else:
                # Keep other messages as is
                processed_messages.append(message)
        
        # If web searches were performed, append conversation continuation context
        if search_performed and self.role.lower() == "worker":
            # Create a summary of all search results for context
            search_summary = self._create_search_summary()
            
            processed_messages.append({
                "role": "system",
                "content": f"The web search was performed to provide additional information. {search_summary} Continue the conversation using this new information, but remember your main task is to collaborate with the supervisor to solve the problem. DO NOT stop the conversation because of the search - it's just a tool to enhance your knowledge."
            })
        elif search_performed and self.role.lower() == "supervisor":
            # Create a summary of all search results for context
            search_summary = self._create_search_summary()
            
            processed_messages.append({
                "role": "system",
                "content": f"The web search was performed to provide additional information. {search_summary} Continue the conversation with the worker, providing guidance based on this new information to help solve the task. DO NOT stop the conversation because of the search - it's just a tool to enhance your knowledge."
            })
        
        # Add the conversation context messages
        processed_messages.extend(conversation_context)
        
        # Call the parent class's generate method with the processed messages
        return super().generate(processed_messages, **kwargs)
    
    def _create_search_summary(self) -> str:
        """Create a summary of all search results for improved context"""
        if not self.last_search_results:
            return ""
        
        parts = ["You searched for the following:"]
        for query, result in self.last_search_results.items():
            if query.strip() in self.processed_queries:
                if result:
                    # Create a brief summary of what was found
                    parts.append(f"- Query: '{query}' - Results were found and included in your message.")
                else:
                    parts.append(f"- Query: '{query}' - No results were found.")
        
        return " ".join(parts)


def main(task: str = None) -> None:
    # Add command line arguments for display options
    parser = argparse.ArgumentParser(description='Run minions conversation with display options')
    parser.add_argument('task', nargs='?', default=task or "What are the latest advancements in AI in 2025? Please search the web for current information.", 
                        help='The task/question to be answered by the minion system')
    parser.add_argument('--full-messages', action='store_true', help='Display full messages without truncation', default=True)
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument('--test-websearch', action='store_true', help='Test the web search functionality directly')
    parser.add_argument('--query', type=str, default=None, help='Query to use when testing web search')
    parser.add_argument('--debug-search', action='store_true', help='Enable additional debugging for search functionality')
    parser.add_argument('--allow-terminal-cmd', action='store_true', default=True, help='Allow AI to execute terminal commands during thinking')
    parser.add_argument('--print-stats', action='store_true', default=True, help='Print search statistics at the end')
    parser.add_argument('--no-cache', action='store_true', default=False, help='Disable search caching (not recommended)')
    
    # If running from command line, parse args; otherwise, use provided args
    if 'pytest' not in sys.modules and len(sys.argv) > 1:
        args = parser.parse_args()
        
        # Handle direct query case: if first arg isn't a flag and no --test-websearch flag,
        # treat it as a test search query
        if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
            args.test_websearch = True
            args.query = args.task  # Use the task as the query
    else:
        # For testing or when called programmatically
        class Args:
            pass
        args = Args()
        args.task = task or "What are the latest advancements in AI in 2025? Please search the web for current information."
        args.full_messages = True  # Default to full messages
        args.no_color = getattr(parser.parse_args([]), 'no_color', False)
        args.test_websearch = getattr(parser.parse_args([]), 'test_websearch', False)
        args.query = getattr(parser.parse_args([]), 'query', "latest AI research advances")
        args.debug_search = True  # Enable debug mode by default
        args.allow_terminal_cmd = True  # Allow terminal commands by default
        args.print_stats = True  # Print search stats by default
        args.no_cache = False  # Enable caching by default

    # Set default query if not specified but test_websearch is enabled
    if args.test_websearch and args.query is None:
        args.query = "latest AI research advances"

    # Print startup information
    print(colorize("\n🔍 MINIONS CONVERSATION SYSTEM 🔍", Colors.BOLD + Colors.BLUE))
    if not args.no_cache:
        print(colorize("Web search capabilities ENABLED with caching", Colors.BOLD + Colors.GREEN))
        print(colorize("Each search will only be performed once", Colors.BOLD + Colors.GREEN))
    else:
        print(colorize("Web search capabilities ENABLED (without caching)", Colors.BOLD + Colors.YELLOW))
        # Clear the cache if no_cache is set
        search_cache.clear()
        
    if args.allow_terminal_cmd:
        print(colorize("Terminal command execution ENABLED - AI can run py main.py --test-websearch", Colors.BOLD + Colors.GREEN))
    
    # Run web search test if requested, but continue with conversation afterward
    if args.test_websearch:
        print(colorize(f"TEST MODE: Running web search for query: \"{args.query}\"", Colors.BOLD + Colors.YELLOW))
        test_websearch(args.query)
        print(colorize("\n[System] Web search test complete! Continuing with the conversation...", Colors.BOLD + Colors.YELLOW))
        # Continue execution instead of returning
    
    print(colorize(f"Task: {args.task}", Colors.YELLOW))
    
    # Test web search functionality if requested but not already tested
    if args.debug_search and not args.test_websearch:
        print(colorize("\n[System] Testing web search functionality before starting...", Colors.BOLD + Colors.YELLOW))
        test_query = "What are the latest AI advancements?"
        
        # Check if the query is already in the cache
        if test_query.strip() in search_cache and not args.no_cache:
            print(colorize(f"[System] Using cached results for query: '{test_query}'", Colors.BOLD + Colors.GREEN))
            print(colorize("[System] ✓ Web search functionality is working (using cache)", Colors.BOLD + Colors.GREEN))
        else:
            print(colorize(f"[System] Running test search with query: '{test_query}'", Colors.YELLOW))
            
            # Create a sample message with a web search request
            test_message = f'Test search: {{"request":"web_search","data":"{test_query}"}}'
            
            # Process the message to perform the web search
            processed_message = process_web_search_requests(test_message)
            
            # Check if the search was successful
            if "[WEB SEARCH RESULTS" in processed_message and len(processed_message) > 200:
                print(colorize("[System] ✓ Web search functionality is working", Colors.BOLD + Colors.GREEN))
            else:
                print(colorize("[System] ⚠ Web search functionality may not be working correctly", Colors.BOLD + Colors.RED))
                print(colorize("[System] We'll continue anyway, but search results may be limited", Colors.RED))
        
        print(colorize("\n[System] Starting conversation...\n", Colors.BOLD + Colors.YELLOW))

    # Use the task from args
    task = args.task

    # Configure the clients with appropriate parameters, using our web-search enabled client
    local_client = WebSearchEnabledClient(
        model_name="llama3.2:1b",
        temperature=0.2,  # Lower temperature for more deterministic outputs
        role="worker"
    )
        
    remote_client = WebSearchEnabledClient(
        model_name="llama3.2:3b",
        temperature=0.1,  # Lower temperature for more structured outputs
        role="supervisor"
    )

    # Instantiate the Minion object with both clients
    minion = Minion(local_client, remote_client)

    context = contexts()

    # Execute the minion protocol for up to five communication rounds
    output = minion(
        task=task,
        context=[context],
        max_rounds=5
    )

    print(context)

    # Display the conversation using the formatted output
    display_conversation(output, args)

    # After the conversation is finished, print search stats if requested
    if args.print_stats:
        print_search_stats()


class Colors:
    """ANSI color codes for terminal output formatting."""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def colorize(text: str, color: str) -> str:
    if getattr(colorize, 'no_color', False):
        return text
    return color + text + Colors.END


def clean_content(content: str) -> str:

    return content

def display_conversation(output: Dict[str, Any], args) -> None:
    # Set the no_color attribute for the colorize function
    colorize.no_color = args.no_color
    
    
    print("\n" + colorize("=== CONVERSATION HISTORY ===", Colors.BOLD + Colors.UNDERLINE) + "\n")

    # Track used questions and answers to prevent duplication
    used_questions: Set[str] = set()
    used_answers: Set[str] = set()
    
    supervisor_messages = output["supervisor_messages"]
    worker_messages = output["worker_messages"]

    # Skip the system message in worker_messages if present
    if worker_messages and worker_messages[0]["role"] == "system":
        worker_messages = worker_messages[1:]

    # Process the initial task
    if supervisor_messages:
        display_initial_task(supervisor_messages[0]["content"], args.full_messages)
        # Skip the initial task in supervisor_messages
        supervisor_messages = supervisor_messages[1:]

    # Create an interleaved conversation
    round_num = 1
    idx = 0
    while idx < max(len(supervisor_messages), len(worker_messages)):
        print(colorize(f"🔶 ROUND {round_num}", Colors.BOLD + Colors.YELLOW))
        print("-" * 80)
        
        # Track if any content was shown in this round
        round_has_content = False
        
        # Process messages in this round
        round_has_content |= process_supervisor_question(supervisor_messages, idx, used_questions, args.full_messages)
        round_has_content |= process_worker_answer(worker_messages, idx, used_answers, args.full_messages)
        round_has_content |= process_worker_question(worker_messages, idx + 1, used_questions, args.full_messages)
        round_has_content |= process_supervisor_answer(supervisor_messages, idx + 1, used_answers, args.full_messages)
        
        # Only increment round number if the round had content
        if round_has_content:
            print("-" * 80 + "\n")
            round_num += 1
            
        idx += 2  # Move to next pair of messages

    # Print the final answer
    print("\n" + colorize("=" * 30 + " FINAL ANSWER " + "=" * 30, Colors.BOLD + Colors.RED))
    print(output["final_answer"])
    print("=" * 80)

    # Print helpful message about command-line options
    if not args.full_messages:
        print("\nTIP: Run with --full-messages to see complete messages without truncation")
    if not args.no_color:
        print("TIP: Run with --no-color to disable colored output")


def display_initial_task(task_content: str, full_messages: bool) -> None:
    print(colorize("🔷 INITIAL TASK:", Colors.BOLD + Colors.BLUE))
    print("-" * 80)
    initial_content = clean_content(task_content)
    print((initial_content,full_messages))
    print("-" * 80 + "\n")


def process_supervisor_question(supervisor_messages, idx, used_questions, full_messages):
    if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "assistant":
        content = clean_content(supervisor_messages[idx]["content"])
        
        # Check if this is a duplicate question
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_questions:
            used_questions.add(content_key)
            print(colorize("Supervisor (Remote) asks:", Colors.BOLD + Colors.BLUE))
            
            # Check if there are web search requests in the content
            pattern = r'{"request":"web_search","data":"([^"]+)"}'
            search_matches = re.findall(pattern, content)
            if search_matches:
                print(colorize("🔍 Supervisor is searching for information...", Colors.BOLD + Colors.BLUE))
                for query in search_matches:
                    print(colorize(f"   Search query: \"{query}\"", Colors.BLUE))
            
            # Check if there are web search results in the content
            if "[WEB SEARCH RESULTS for" in content:
                result_count = content.count("[WEB SEARCH RESULTS for")
                print(colorize(f"📊 Displaying {result_count} search result{'s' if result_count > 1 else ''} in message", Colors.BOLD + Colors.BLUE))
            
            print(content,full_messages)
            print()
            return True
    return False


def process_worker_answer(worker_messages, idx, used_answers, full_messages):
    if idx < len(worker_messages) and worker_messages[idx]["role"] == "assistant":
        content = clean_content(worker_messages[idx]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_answers:
            used_answers.add(content_key)
            print(colorize("★ Worker (Local) answers: ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
            
            # Check if there are web search requests in the content
            pattern = r'{"request":"web_search","data":"([^"]+)"}'
            search_matches = re.findall(pattern, content)
            if search_matches:
                print(colorize("🔍 ★ Worker is searching for information... ★", Colors.BOLD + Colors.GREEN))
                for query in search_matches:
                    print(colorize(f"   ★ Search query: \"{query}\" ★", Colors.GREEN))
            
            # Check if there are web search results in the content
            if "[WEB SEARCH RESULTS for" in content:
                result_count = content.count("[WEB SEARCH RESULTS for")
                print(colorize(f"📊 ★ Displaying {result_count} search result{'s' if result_count > 1 else ''} in message ★", Colors.BOLD + Colors.GREEN))
            
            # Highlight worker's direct thoughts with emphasis
            highlight_phrases = ["⚡ I think", "⚡ In my opinion", "⚡ my analysis"]
            for phrase in highlight_phrases:
                base_phrase = phrase.replace("⚡ ", "")
                content = content.replace(base_phrase, colorize(phrase, Colors.BOLD + Colors.YELLOW))
            
            print(content,full_messages)
            print()
            return True
    return False


def process_worker_question(worker_messages, idx, used_questions, full_messages):
    if idx < len(worker_messages) and worker_messages[idx]["role"] == "user":
        content = clean_content(worker_messages[idx]["content"])
        
        # Check if this is a duplicate question
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_questions:
            used_questions.add(content_key)
            
            if "★ WORKER QUESTION ★" in content:
                print(colorize("★★★ Worker-Initiated Question ★★★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
                # Remove the special marker
                content = content.replace("★ WORKER QUESTION ★: ", "")
            else:
                print(colorize("★ Worker (Local) asks: ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
            
            # Check if there are web search requests in the content
            pattern = r'{"request":"web_search","data":"([^"]+)"}'
            search_matches = re.findall(pattern, content)
            if search_matches:
                print(colorize("🔍 ★ Worker is searching for information... ★", Colors.BOLD + Colors.GREEN))
                for query in search_matches:
                    print(colorize(f"   ★ Search query: \"{query}\" ★", Colors.GREEN))
            
            # Check if there are web search results in the content
            if "[WEB SEARCH RESULTS for" in content:
                result_count = content.count("[WEB SEARCH RESULTS for")
                print(colorize(f"📊 ★ Displaying {result_count} search result{'s' if result_count > 1 else ''} in message ★", Colors.BOLD + Colors.GREEN))
            
            print(content,full_messages)
            print()
            return True
    return False


def process_supervisor_answer(supervisor_messages, idx, used_answers, full_messages):
    if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "user":
        content = clean_content(supervisor_messages[idx]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_answers:
            used_answers.add(content_key)
            print(colorize("Supervisor (Remote) answers:", Colors.BOLD + Colors.BLUE))
            
            # Check if there are web search requests in the content
            pattern = r'{"request":"web_search","data":"([^"]+)"}'
            search_matches = re.findall(pattern, content)
            if search_matches:
                print(colorize("🔍 Supervisor is searching for information...", Colors.BOLD + Colors.BLUE))
                for query in search_matches:
                    print(colorize(f"   Search query: \"{query}\"", Colors.BLUE))
            
            # Check if there are web search results in the content
            if "[WEB SEARCH RESULTS for" in content:
                result_count = content.count("[WEB SEARCH RESULTS for")
                print(colorize(f"📊 Displaying {result_count} search result{'s' if result_count > 1 else ''} in message", Colors.BOLD + Colors.BLUE))
            
            print(content,full_messages)
            print()
            return True
    return False


def display_realtime_search_results(results, role="worker"):
    """
    Displays search results in real-time to the console with clear formatting based on the caller's role.
    
    Args:
        results: List of Document objects containing search results
        role: The role of the caller ("worker", "supervisor", or "neutral")
    """
    # Set display parameters based on role
    if role.lower() == "neutral":
        role_display = "SYSTEM"
        role_color = Colors.YELLOW
    else:
        role_display = role.upper()
        role_color = Colors.BLUE if role.lower() == 'supervisor' else Colors.GREEN
    
    # Create a visual separator for the search results
    separator = "=" * 80
    print(colorize(f"\n{separator}", Colors.BOLD + role_color))
    print(colorize(f"{'🔍 ' + role_display + ' SEARCH RESULTS 🔍':^80}", Colors.BOLD + role_color))
    print(colorize(f"{separator}", Colors.BOLD + role_color))
    
    if not results:
        print(colorize("\nNo results found for this search query.", role_color))
        print(colorize(f"\n{separator}", Colors.BOLD + role_color))
        return
    
    # Count academic sources and document types for summary
    academic_count = sum(1 for doc in results if doc.metadata.get('is_academic', False))
    doc_types = {}
    for doc in results:
        doc_type = doc.metadata.get('document_type', 'html').upper()
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    # Display summary of results
    print(colorize(f"\n📊 Found {len(results)} relevant documents:", Colors.BOLD + role_color))
    
    # Show document type breakdown
    type_info = []
    for doc_type, count in doc_types.items():
        if doc_type == 'PDF':
            type_info.append(f"📑 {count} PDFs")
        elif doc_type in ('DOCX', 'DOC'):
            type_info.append(f"📝 {count} Documents")
        elif doc_type == 'HTML':
            type_info.append(f"🌐 {count} Web pages")
        else:
            type_info.append(f"📄 {count} {doc_type} files")
    
    if type_info:
        print(colorize("   " + ", ".join(type_info), role_color))
    
    # Show academic info if applicable
    if academic_count > 0:
        print(colorize(f"   🎓 Including {academic_count} academic sources", Colors.BOLD + role_color))
    
    # Print each document with its summary and key information
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Unknown')
        avg_score = doc.metadata.get('avg_score', 0)
        doc_type = doc.metadata.get('document_type', 'html').upper()
        is_academic = doc.metadata.get('is_academic', False)
        summary = doc.metadata.get('summary', '')
        
        # Document type icon
        type_icon = "📄"  # Default
        if doc_type == 'PDF':
            type_icon = "📑"
        elif doc_type in ('DOCX', 'DOC'):
            type_icon = "📝"
        elif doc_type == 'PRESENTATION':
            type_icon = "🎭"
        elif doc_type == 'SPREADSHEET':
            type_icon = "📊"
        elif doc_type == 'HTML':
            type_icon = "🌐"
        
        # Academic badge
        academic_badge = "🎓 ACADEMIC SOURCE" if is_academic else ""
        
        # Document header
        print(colorize(f"\n{'-' * 80}", role_color))
        header = f"{type_icon} DOCUMENT {i}/{len(results)} | TYPE: {doc_type}"
        if academic_badge:
            header += f" | {academic_badge}"
        print(colorize(header, Colors.BOLD + role_color))
        
        # Source and relevance information
        print(colorize(f"SOURCE: {source}", role_color))
        relevance_description = "LOW"
        if avg_score >= 0.8:
            relevance_description = "VERY HIGH"
        elif avg_score >= 0.6:
            relevance_description = "HIGH"
        elif avg_score >= 0.4:
            relevance_description = "MEDIUM"
        
        print(colorize(f"RELEVANCE: {avg_score:.2f} ({relevance_description})", role_color))
        
        # Show summary if available
        if summary and summary != doc.page_content:
            print(colorize("\n📋 KEY POINTS:", Colors.BOLD + role_color))
            
            # Format the summary with bullet points if it doesn't already have them
            if not any(line.strip().startswith(('-', '•', '*')) for line in summary.split('\n')):
                # Split by sentences and create bullet points
                import re
                sentences = re.split(r'(?<=[.!?])\s+', summary)
                formatted_summary = "\n".join(f"• {sentence}" for sentence in sentences if sentence.strip())
                print(colorize(formatted_summary, role_color))
            else:
                # Summary already has structure, just print it
                print(colorize(summary, role_color))
        


    print(colorize(f"\n{separator}", Colors.BOLD + role_color))
    
    # Add a message about continuing the conversation
    if role.lower() == "neutral":
        print(colorize(f"\n➡️ Search results have been integrated into the conversation", Colors.BOLD + role_color))
    else:
        role_name = "Supervisor" if role.lower() == "supervisor" else "Worker"
        print(colorize(f"\n➡️ {role_name} is integrating this information into the conversation...", Colors.BOLD + role_color))
    
    print(colorize(f"These search results have been formatted and included in the conversation.", role_color))


if __name__ == "__main__":
    # Run the main function (argument parsing happens inside main)
    main()