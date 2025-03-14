from minions.minion import Minion
import argparse
import sys
from main_program.contxtss import contexts
from main_program.website_simulations import  search_cache,process_web_search_requests, print_search_stats,test_websearch
from main_program.colors import colorize, Colors
from main_program.WebSearchEnabledClients import WebSearchEnabledClient
from main_program.rounds import display_conversation
def main(task: str = None) -> None:
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
    if 'pytest' not in sys.modules and len(sys.argv) > 1:
        args = parser.parse_args()
        if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
            args.test_websearch = True
            args.query = args.task  
    else:
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
    if args.test_websearch and args.query is None:
        args.query = "latest AI research advances"
    print(colorize("\n🔍 MINIONS CONVERSATION SYSTEM 🔍", Colors.BOLD + Colors.BLUE))
    if not args.no_cache:
        print(colorize("Web search capabilities ENABLED with caching", Colors.BOLD + Colors.GREEN))
        print(colorize("Each search will only be performed once", Colors.BOLD + Colors.GREEN))
    else:
        print(colorize("Web search capabilities ENABLED (without caching)", Colors.BOLD + Colors.YELLOW))
        search_cache.clear()
        
    if args.allow_terminal_cmd:
        print(colorize("Terminal command execution ENABLED - AI can run py main.py --test-websearch", Colors.BOLD + Colors.GREEN))
    if args.test_websearch:
        print(colorize(f"TEST MODE: Running web search for query: \"{args.query}\"", Colors.BOLD + Colors.YELLOW))
        test_websearch(args.query)
        print(colorize("\n[System] Web search test complete! Continuing with the conversation...", Colors.BOLD + Colors.YELLOW))
    
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


    local_client = WebSearchEnabledClient(
        model_name="llama3.2:1b",
        temperature=0.2,  
        role="worker"
    )
        
    remote_client = WebSearchEnabledClient(
        model_name="llama3.2:3b",
        temperature=0.1,  
        role="supervisor"
    )
    minion = Minion(local_client, remote_client)
    context = contexts()

    output = minion(
        task=task,
        context=[context],
        max_rounds=5
    )

    print(context)
    display_conversation(output, args)

    if args.print_stats:
        print_search_stats()

if __name__ == "__main__":
    main()