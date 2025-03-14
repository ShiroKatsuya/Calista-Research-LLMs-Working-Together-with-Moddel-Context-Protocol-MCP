from websitetools import search_and_load
from main_program.colors import colorize, Colors
import re
import time
from main_program.execute_terminal_commands import execute_terminal_command
search_cache = {}
# Search statistics
search_stats = {
    "total_requests": 0,    # Total number of search requests
    "actual_searches": 0,   # Number of unique searches actually performed
    "cached_hits": 0,       # Number of times the cache was used
}

def clean_content(content: str) -> str:

    return content
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
                    display_realtime_search_results(search_result_data, "neutral")
                    
                    # Create a formatted representation of the results for the message
                    formatted_results = format_search_results_for_message(structured_results, query)
                else:
                    formatted_results = f'\n\n ➡️ {colorize(f"[WEB SEARCH RESULTS for \"{query}\"]", Colors.YELLOW)}\n{colorize("No results found.", Colors.YELLOW)}\n ➡️ {colorize("[END OF SEARCH RESULTS]", Colors.YELLOW)}\n\n'
            else:
                formatted_results = f'\n\n ➡️ {colorize(f"[WEB SEARCH RESULTS for \"{query}\"]", Colors.YELLOW)}\n{colorize("[ERROR: Empty search query]", Colors.YELLOW)}\n ➡️ {colorize("[END OF SEARCH RESULTS]", Colors.YELLOW)}\n\n'
                search_result_data = []
        except Exception as e:
            # Handle any exceptions during the search
            import traceback
            error_message = f"[ERROR during web search: {str(e)}]\n{traceback.format_exc()}"
            print(f"{colorize(f'[Debug] Search error: {str(e)}', Colors.RED)}")
            formatted_results = f'\n\n ➡️ {colorize(f"[WEB SEARCH RESULTS for \"{query}\"]", Colors.YELLOW)}\n{colorize(error_message, Colors.YELLOW)}\n ➡️ {colorize("[END OF SEARCH RESULTS]", Colors.YELLOW)}\n\n'
            search_result_data = []
        
        # Replace the search request with the results
        message = message.replace(f'{{"request":"web_search","data":"{query}"}}', formatted_results)
    
    return message
def display_realtime_search_results(results, role="worker"):
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
        
        # Show a preview of the content (truncated for readability)
        print(colorize("\n📃 CONTENT PREVIEW:", Colors.BOLD + role_color))
        content = doc.page_content
        max_preview_length = 500  # Reasonable preview length
        if len(content) > max_preview_length:
            preview = content[:max_preview_length] + "... [content continues]"
        else:
            preview = content
        
        # Clean and format the preview
        preview = preview.replace('\n\n', '\n').strip()
        print(colorize(preview, role_color))
    
    print(colorize(f"\n{separator}", Colors.BOLD + role_color))
    
    # Add a message about continuing the conversation
    if role.lower() == "neutral":
        print(colorize(f"\n ➡️ Search results have been integrated into the conversation", Colors.BOLD + role_color))
    else:
        role_name = "Supervisor" if role.lower() == "supervisor" else "Worker"
        print(colorize(f"\n ➡️ {role_name} is integrating this information into the conversation...", Colors.BOLD + role_color))
    
    print(colorize(f"These search results have been formatted and included in the conversation.", role_color))


def format_search_results_for_message(structured_results, query):
    """
    Formats structured search results into a clean text representation for inclusion in messages.
    
    Args:
        structured_results: List of dictionaries containing document data
        query: The original search query
        
    Returns:
        Formatted string representation of search results
    """
    if not structured_results:
        return f"\n\n ➡️ ' ' {colorize(f'[WEB SEARCH RESULTS for "{query}"]', Colors.YELLOW)}\nNo results found.\n ➡️ ' ' {colorize('[END OF SEARCH RESULTS]', Colors.YELLOW)}\n\n"
    
    # Start building the formatted results
    result_parts = [f"\n\n ➡️ ' ' {colorize(f"[WEB SEARCH RESULTS for \"{query}\"]", Colors.YELLOW)}\n"]
    result_parts.append(f"\n\n ➡️ ' ' Found {len(structured_results)} relevant documents.\n")
    
    # Count academic sources
    academic_count = sum(1 for doc in structured_results if doc["is_academic"])
    if academic_count > 0:
        result_parts.append(f"\n\n ➡️ Including {academic_count} academic sources.\n")
    

    
    # End of results
    result_parts.append(f"\n ➡️ ' ' {colorize('[END OF SEARCH RESULTS]', Colors.YELLOW)}\n")
    result_parts.append(f"\n ➡️ ' ' {colorize('Using this information, I will continue our conversation. The search results are just additional context to enhance our discussion...', Colors.YELLOW)}\n\n")
    return "\n".join(result_parts)



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
    
    message = f'\n\n ➡️ {colorize(f"I need information about {query}. {{\"request\":\"web_search\",\"data\":\"{query}\"}}", Colors.YELLOW)}'
    
    print(colorize("\nOriginal message (string format):", Colors.YELLOW))
    print(colorize(f"\n ➡️ {message}", Colors.YELLOW))
    
    # Process the message to perform the web search
    print(colorize("\nProcessing web search request (string format)...", Colors.YELLOW))
    processed_message = process_web_search_requests(message)
    
    print(processed_message)
    
    # Show how the conversation would continue
    print(colorize("\nSimulated conversation AFTER search:", Colors.BOLD + Colors.YELLOW))
    print(colorize("Worker: Based on my search, I found several important advancements in AI. Recent papers mention...", Colors.GREEN))
    print(colorize("Supervisor: Great information. Let's use this to address the specific question about...", Colors.BLUE))
    
    # Test 3: Test dictionary-based request handling (simulating model error)
    print(colorize("\n==== TESTING DICTIONARY REQUEST HANDLING ====", Colors.BOLD + Colors.YELLOW))
    
    # Create a message with a dictionary as content instead of a string
    dict_message = {
        "role": "assistant",
        "content": {"request": "web_search", "data": query}
    }
    
    messages = [{"role": "system", "content": "You are a helpful assistant"}, dict_message]
    
    print("\nOriginal message (dict format):")
    print(dict_message)
    processed_messages = []
    for message in messages:
        if message["role"] in ["assistant", "user"]:
            if isinstance(message["content"], dict) and "request" in message["content"] and message["content"]["request"] == "web_search":
                # This is the key part that handles the dictionary content case
                print("\nDetected dictionary-based web search request, converting to string format...")
                query = message["content"].get("data", "")
                if query:
                    message["content"] = f'{{"request":"web_search","data":"{query}"}}'
                    print(f"Converted content to: {message['content']}")
                    
                    # Now process the web search using the process_web_search_requests function
                    print("\nProcessing the converted web search request...")
                    message["content"] = process_web_search_requests(message["content"])
                else:
                    message["content"] = "Error: Invalid search request - missing query data"
            
            processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    print("\nProcessed messages (after handling dict format):")
    
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