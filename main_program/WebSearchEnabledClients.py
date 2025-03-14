from main_program.website_simulations import colorize,Colors,display_thinking_state,process_web_search_requests
import re
from minions.clients.ollama import OllamaClient
from typing import Dict, Any, Set, List
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
                            search_pattern = re.escape(f' ➡️ {colorize(f"[WEB SEARCH RESULTS for \"{query}\"]", Colors.YELLOW)}') + r'(.*?)' + re.escape(f' ➡️{colorize("[END OF SEARCH RESULTS]", Colors.YELLOW)}')
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
