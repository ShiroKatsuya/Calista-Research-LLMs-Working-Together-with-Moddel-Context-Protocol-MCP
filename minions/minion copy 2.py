from typing import List, Dict, Any, Optional, Set, Tuple, Callable
import json
import re
import os
import time
from datetime import datetime

from minions.clients import OpenAIClient, TogetherClient
from minions.clients.base import BaseClient

from minions.prompts.minion import (
    SUPERVISOR_CONVERSATION_PROMPT,
    SUPERVISOR_FINAL_PROMPT,
    SUPERVISOR_INITIAL_PROMPT,
    WORKER_SYSTEM_PROMPT,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_FINAL,
    WORKER_PRIVACY_SHIELD_PROMPT,
    REFORMAT_QUERY_PROMPT,
)
from minions.usage import Usage
from minions.utils import escape_newlines_in_strings, extract_json, clean_json_string, aggressive_json_repair, apply_privacy_shield

# Import Colors class for terminal coloring
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
    """Add color to text for terminal output."""
    # Allow disabling colors via function attribute
    if getattr(colorize, "no_color", False):
        return text
    return f"{color}{text}{Colors.END}"

def _escape_newlines_in_strings(json_str: str) -> str:
    # This regex naively matches any content inside double quotes (including escaped quotes)
    # and replaces any literal newline characters within those quotes.
    # was especially useful for anthropic client
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    # Handle empty strings or None values
    if not text or not text.strip():
        return {
            "decision": "provide_final_answer",
            "answer": "Empty response received",
            "message": "Could not process response as JSON - empty string provided"
        }
        
    # First, try to find JSON in code blocks
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    
    # Then look for standalone JSON objects - looking for the largest JSON object
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
    
    # Sort bracket matches by length to find the most complete JSON object
    if bracket_matches:
        bracket_matches.sort(key=lambda m: len(m.group(0)), reverse=True)

    # Try code blocks first (most common format from LLMs)
    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    # Fall back to bracket matching
    elif bracket_matches:
        json_str = bracket_matches[0].group(0)  # Use the largest JSON object found
    # If no matches, use the whole text
    else:
        json_str = text

    # Clean up the JSON string before parsing
    json_str = _clean_json_string(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try aggressive repair
        try:
            cleaned = _aggressive_json_repair(json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: try to build a valid JSON object from scratch
            decision_match = re.search(r'"decision"\s*:\s*"([^"]*)"', json_str)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', json_str)
            message_match = re.search(r'"message"\s*:\s*"([^"]*)"', json_str)
            
            result = {
                "decision": decision_match.group(1) if decision_match else "provide_final_answer",
                "answer": answer_match.group(1) if answer_match else f"Failed to parse response: {text}...",
                "message": message_match.group(1) if message_match else "Could not process response as JSON"
            }
            return result


def _clean_json_string(json_str: str) -> str:
    """Clean up a JSON string to improve parsing success."""
    # Escape newlines in strings
    json_str = _escape_newlines_in_strings(json_str)
    
    # Replace Unicode quotes with standard quotes
    json_str = json_str.replace('"', '"').replace('"', '"')
    json_str = json_str.replace("'", "'").replace("'", "'")
    
    # Replace single quotes with double quotes (for keys and string values)
    # This is tricky and may not work for all cases
    # json_str = re.sub(r'(\w+)\':', r'\1":', json_str)  # Fix keys
    # json_str = re.sub(r':\'([^\']+)\'', r':"\1"', json_str)  # Fix simple values
    
    # Remove trailing commas in arrays and objects
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    return json_str


def _aggressive_json_repair(json_str: str) -> str:
    """More aggressive JSON repair for when standard cleaning fails."""
    # Fix unescaped quotes within strings
    fixed = ""
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            fixed += char
            escape_next = False
            continue
            
        if char == '\\':
            fixed += char
            escape_next = True
            continue
            
        if char == '"' and not in_string:
            in_string = True
            fixed += char
            continue
            
        if char == '"' and in_string and not escape_next:
            in_string = False
            fixed += char
            continue
            
        # Handle special characters in strings
        if in_string and char in ['\n', '\r', '\t']:
            if char == '\n':
                fixed += '\\n'
            elif char == '\r':
                fixed += '\\r'
            elif char == '\t':
                fixed += '\\t'
            continue
            
        fixed += char
    
    # Ensure strings are properly closed
    if in_string:
        fixed += '"'
    
    # Replace single quotes around keys with double quotes
    fixed = re.sub(r"'([^']+)':", r'"\1":', fixed)
    
    # Fix trailing commas in objects and arrays
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    
    # Ensure the JSON object is properly closed
    fixed = fixed.strip()
    if not fixed.endswith('}') and fixed.startswith('{'):
        fixed += '}'
    if not fixed.startswith('{') and fixed.endswith('}'):
        fixed = '{' + fixed
        
    return fixed


class Minion:
    """
    Minion class for orchestrating conversations between a local and remote LLM client.
    """
    
    def __init__(
        self,
        local_client: Optional[BaseClient] = None,
        remote_client: Optional[BaseClient] = None,
        max_rounds: Optional[int] = 5,
        callback: Optional[Callable] = None,
        log_dir: str = "minion_logs",
    ):
   
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds or 5  # Default to 5 rounds if None
        self.callback = callback
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Track used questions and answers to avoid duplicates
        self.used_questions: Set[str] = set()
        self.used_answers: Set[str] = set()
        
        # Cache for expensive operations
        self._session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds: Optional[int] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        logging_id: Optional[str] = None,  # this is the name/id to give to the logging .json file
        is_privacy: bool = False,
    ) -> Dict[str, Any]:
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files
            is_privacy: Whether to use privacy shield

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """
        start_time = time.time()
        
        if max_rounds is None:
            max_rounds = self.max_rounds
            
        # Reset tracking of used questions and answers for this run
        self.used_questions = set()
        self.used_answers = set()

        # Join context sections
        merged_context = "\n\n".join(context)

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": merged_context,
            "conversation": [],
            "generated_final_answer": "",
            "metadata": {
                "timestamp": self._session_timestamp,
                "document_metadata": doc_metadata or {},
            }
        }

        # Track usage statistics
        local_usage = 0
        remote_usage = 0

        # Setup messages and handle privacy if needed
        supervisor_messages, worker_messages = self._setup_initial_messages(
            task, merged_context, is_privacy, conversation_log
        )

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        # Get first supervisor question
        supervisor_response, supervisor_usage = self._get_supervisor_response(supervisor_messages)
        remote_usage += supervisor_usage
        
        # Add supervisor's first question to the conversation
        self._add_supervisor_question(
            supervisor_response, supervisor_messages, conversation_log
        )

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Main conversation loop
        final_answer = self._run_conversation_rounds(
            task, 
            max_rounds, 
            supervisor_messages, 
            worker_messages, 
            conversation_log,
            local_usage,
            remote_usage,
            is_privacy
        )

        # Save the conversation log
        log_filename = self._save_conversation_log(conversation_log, logging_id)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Return results
        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "local_usage": local_usage,
            "remote_usage": remote_usage,
            "log_path": log_filename,
            "execution_time": execution_time,
        }
        
    def _setup_initial_messages(
        self, 
        task: str, 
        context: str, 
        is_privacy: bool,
        conversation_log: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Set up initial messages for both models, handling privacy if needed."""
        if is_privacy:
            # Handle privacy-aware processing
            from minions.utils.pii_extraction import PIIExtractor
            
            # Extract PII
            pii_extractor = PIIExtractor()
            pii_extracted = pii_extractor.extract_pii(context)
            query_pii_extracted = pii_extractor.extract_pii(task)
            
            # Reformat query with PII awareness
            reformat_query_task = REFORMAT_QUERY_PROMPT.format(
                query=task, pii_extracted=str(query_pii_extracted)
            )
            
            # Clean PII from query
            reformatted_task, _, _ = self.local_client.chat(
                messages=[{"role": "user", "content": reformat_query_task}]
            )
            pii_reformatted_task = reformatted_task[0]
            
            # Log the reformatted task
            if self.callback:
                self.callback("worker", f"**PII Reformated Task:**\n{pii_reformatted_task}")
                
            # Initialize message histories with PII awareness
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(task=pii_reformatted_task),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                }
            ]
            
            # Add to conversation log
            conversation_log["conversation"].append({
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=pii_reformatted_task),
                "output": None,
            })
        else:
            # Standard initialization without privacy
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                }
            ]
            
            # Add to conversation log
            conversation_log["conversation"].append({
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                "output": None,
            })
            
        return supervisor_messages, worker_messages
    

    
    def _get_supervisor_response(self, supervisor_messages: List[Dict[str, str]]):
        """Get response from the supervisor model with appropriate handling for different client types."""
        try:
            # Set up streaming for supervisor response
            buffer = ""
            
            def supervisor_stream_callback(chunk):
                nonlocal buffer
                # Print the chunk immediately
                print(chunk, end="", flush=True)
                buffer += chunk
            
            # Display that the supervisor is thinking
            print(colorize("\nSupervisor (Remote) is thinking...", Colors.BOLD + Colors.BLUE))

            from deep_translator import GoogleTranslator 
            
            # Check if client supports response format parameter (like OpenAI)
            if hasattr(self.remote_client, "supports_response_format") and self.remote_client.supports_response_format:
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages, 
                    response_format={"type": "json_object"},
                    stream_callback=supervisor_stream_callback
                )
            else:
                supervisor_response, supervisor_usage, _ = self.remote_client.chat(
                    messages=supervisor_messages,
                    stream_callback=supervisor_stream_callback
                )
            
            # Clear the buffer with a newline
            print("\n")

            # Process supervisor response
            
            return supervisor_response, supervisor_usage
        except Exception as e:
            # Log error and return a fallback response
            print(f"Error getting supervisor response: {e}")
            return [f"I encountered an error: {str(e)}. Could you help with this task?"], 0
    
    def _add_supervisor_question(
        self, 
        supervisor_response: List[str], 
        supervisor_messages: List[Dict[str, str]],
        conversation_log: Dict[str, Any],
        already_displayed: bool = True  # Default to True since we're streaming
    ):
        """Add supervisor's question to the messages and log."""
        supervisor_messages.append({"role": "assistant", "content": supervisor_response[0]})
        
        # Add to conversation log
        conversation_log["conversation"].append({
            "user": "remote", 
            "prompt": None, 
            "output": supervisor_response[0]
        })
        
        # If not already displayed via streaming, display now
        if not already_displayed and self.callback:
            self.callback("remote", supervisor_response[0], is_final=False)
        
    def _process_worker_response(
        self,
        worker_response: List[str],
        worker_messages: List[Dict[str, str]],
        conversation_log: Dict[str, Any],
        task: str,
        is_privacy: bool,
        pii_extracted: Optional[str] = None,
        already_displayed: bool = False
    ):
        """Process the worker's response, handling privacy if needed and formatting the output."""
        # Check for response uniqueness
        response_key = worker_response[0].strip()[:100]
        
        if response_key in self.used_answers:
            # Request a more unique response if duplicate detected
            worker_uniqueness_prompt = """Your previous response was too similar to one you've already provided. 
            Please provide a more unique and detailed perspective on the topic that adds new information.
            Be specific and try to approach the question from a different angle. 
            Avoid repeating similar points, phrases, or structures from your previous responses."""
            
            worker_messages.append({"role": "user", "content": worker_uniqueness_prompt})
            
            # If we're streaming, show this is happening
            if not already_displayed:
                print(colorize("Request for a more unique response...", Colors.BOLD + Colors.YELLOW))
            
            # For uniqueness requests, don't stream to avoid confusion
            worker_response, _, _ = self.local_client.chat(
                messages=worker_messages
            )
            
            # Remove the uniqueness prompt to avoid confusion in future exchanges
            worker_messages.pop()
        
        # Add to used answers
        self.used_answers.add(response_key)
        
        if is_privacy:
            # Apply privacy shield
            if self.callback:
                self.callback("worker", f"**_My output (pre-privacy shield):_**\n\n{worker_response[0]}")
                
            original_response = worker_response[0]
            shielded_response = apply_privacy_shield(
                original_response, 
                self.local_client,
                pii_json=pii_extracted
            )
            worker_response = [shielded_response]
            
            if self.callback:
                self.callback("worker", f"**_My output (post-privacy shield):_**\n\n{shielded_response}")
        
        # Emphasize the response
        worker_response[0] = self._emphasize_worker_response(worker_response[0])
        
        # Add to messages and log
        worker_messages.append({"role": "assistant", "content": worker_response[0]})
        
        # Add to conversation log
        conversation_log["conversation"].append({
            "user": "local", 
            "prompt": None, 
            "output": worker_response[0]
        })
        
        # Display the emphasized response if not already displayed
        if not already_displayed and self.callback:
            self.callback("worker", worker_response[0], is_final=False)
        
        return worker_response
    
    def _emphasize_worker_response(self, response: str) -> str:
        """Add emphasis and prefixes to worker responses to distinguish them."""
        # Ensure response has supervisor prefix
        if not response.startswith("@Supervisor:"):
            emphasized_response = "@Supervisor: " + response
        else:
            emphasized_response = response
        
        # Highlight worker's direct thoughts with bold markers
        thought_phrases = [
            "I think", "In my opinion", "my analysis", "From my perspective", 
            "My understanding is", "I believe", "I'd suggest", "I propose",
            "my view", "my interpretation", "I'm thinking", "I disagree", "I agree"
        ]
        
        for phrase in thought_phrases:
            if phrase in emphasized_response:
                emphasized_response = emphasized_response.replace(phrase, f"⚡ {phrase}")
        
        return emphasized_response
    
    def _handle_worker_follow_up(
        self,
        task: str,
        worker_messages: List[Dict[str, str]],
        supervisor_messages: List[Dict[str, str]],
        conversation_log: Dict[str, Any],
        worker_response: List[str]
    ) -> Tuple[bool, int]:
        """Handle worker follow-up questions if any."""
        # Create follow-up prompt
        worker_follow_up_prompt = f"""Based on your last response:
        "{worker_response[0]}"
        
        Do you have any questions you would like to ask the supervisor model? 
        If so, please provide a clear, specific question that will help you better
        understand or solve the task at hand. If you do not have any questions,
        say "No questions at this time."
        
        Keep your question direct and focused on the task: "{task}"
        
        Ensure your question is unique and hasn't been asked before. Ask something that
        will provide new information or a different perspective on the task.
        
        The best questions are those that:
        1. Seek clarification on aspects of the task you're unsure about
        2. Request specific information that would help you provide a better response
        3. Challenge assumptions in a constructive way
        4. Explore perspectives that haven't been considered yet
        5. Show critical thinking about the problem
        
        Remember: Your questions drive the conversation forward, so make them count!
        """
        
        # Get worker's follow-up question
        worker_question_response, worker_follow_up_usage, _ = self.local_client.chat(
            messages=worker_messages + [{"role": "user", "content": worker_follow_up_prompt}]
        )
        
        # Check if worker has a follow-up question and it's not a duplicate
        if not worker_question_response or worker_question_response[0].strip() == "No questions at this time.":
            # No follow-up question
            return False, worker_follow_up_usage
            
        question_key = worker_question_response[0].strip()[:100]
        
        # Check if question is unique
        if question_key in self.used_questions:
            # Question is a duplicate
            return False, worker_follow_up_usage
            
        # Add to used questions
        self.used_questions.add(question_key)
        
        # Format question
        emphasized_question = "★ WORKER QUESTION ★: " + worker_question_response[0]
        
        # Add to conversation log
        conversation_log["conversation"].append({
            "user": "local", 
            "prompt": emphasized_question, 
            "output": None
        })
        
        # Create supervisor prompt
        supervisor_reply_prompt = f"""
        ### Question
        {worker_question_response[0]}
        
        ### Instructions
        Please answer this specific question to help the small language model. Provide clear, 
        helpful information directly related to this question. Remember that the small model
        has access to the context, but may need guidance on how to interpret or use it.
        
        After thinking step-by-step, provide your answer in the following format:
        
        ```json
        {{
            "decision": "request_additional_info",
            "message": "<your answer to the small model's question>"
        }}
        ```
        """
        
        # Add to supervisor messages
        supervisor_messages.append({"role": "user", "content": supervisor_reply_prompt})
        
        # Get supervisor response
        supervisor_reply, supervisor_usage = self._get_supervisor_response(supervisor_messages)
        
        # Add to supervisor messages
        supervisor_messages.append({"role": "assistant", "content": supervisor_reply[0]})
        
        # Update conversation log
        conversation_log["conversation"][-1]["output"] = supervisor_reply[0]
        
        # Process supervisor reply
        try:
            # Try to parse JSON
            supervisor_reply_json = json.loads(supervisor_reply[0])
        except json.JSONDecodeError:
            # Try to extract JSON if parsing fails
            supervisor_reply_json = extract_json(supervisor_reply[0])
        
        # Get supervisor's answer
        supervisor_answer = supervisor_reply_json.get("message", supervisor_reply[0])
        
        # Add answer to worker messages
        worker_messages.append({"role": "user", "content": supervisor_answer})
        
        return True, supervisor_usage + worker_follow_up_usage
        
    def _run_conversation_rounds(
        self,
        task: str,
        max_rounds: int,
        supervisor_messages: List[Dict[str, str]],
        worker_messages: List[Dict[str, str]],
        conversation_log: Dict[str, Any],
        local_usage: int,
        remote_usage: int,
        is_privacy: bool
    ) -> str:
        """Run the main conversation rounds between worker and supervisor."""
        final_answer = None
        pii_extracted = None
        
        # If using privacy, extract PII information
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor
            pii_extractor = PIIExtractor()
            pii_extracted = pii_extractor.extract_pii("\n\n".join(
                [m["content"] for m in worker_messages if m.get("content")]
            ))
        
        # Main conversation loop
        for round_idx in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)
            
            # Set up streaming output for worker responses
            print(colorize("★ Worker (Local) is thinking... ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
            buffer = ""
            
            def worker_stream_callback(chunk):
                nonlocal buffer
                # Print the chunk immediately
                print(chunk, end="", flush=True)
                buffer += chunk
            
            # Call the client with the streaming callback
            worker_response, worker_usage, _ = self.local_client.chat(
                messages=worker_messages, 
                stream_callback=worker_stream_callback
            )
            
            # Clear the buffer by printing a newline
            print("\n")
            
            local_usage += worker_usage
            
            # Process worker response - don't need to display since we already streamed it
            worker_response = self._process_worker_response(
                worker_response, worker_messages, conversation_log, 
                task, is_privacy, pii_extracted,
                already_displayed=True  # Add this flag to indicate it's already been displayed
            )
            
            # Check for worker follow-up questions
            had_followup, followup_usage = self._handle_worker_follow_up(
                task, worker_messages, supervisor_messages, conversation_log, worker_response
            )
            
            local_usage += followup_usage
            
            # Get response from the supervisor
            if self.callback:
                self.callback("remote", None, is_final=False)
                
            supervisor_response, supervisor_usage = self._get_supervisor_response(supervisor_messages)
            remote_usage += supervisor_usage
            
            # Add supervisor's question (it's already displayed since we streamed it)
            self._add_supervisor_question(
                supervisor_response, supervisor_messages, conversation_log,
                already_displayed=True
            )
            
            # Check for final answer in worker's response
            final_answer = self._extract_final_answer(worker_response[0])
            if final_answer:
                # We have a final answer, end the conversation
                conversation_log["generated_final_answer"] = final_answer
                break
                
            # If this isn't the last round and we didn't have a follow-up, get next supervisor question
            if round_idx < max_rounds - 1 and not had_followup:
                # Prepare supervisor prompt
                supervisor_next_prompt = self._create_supervisor_prompt(worker_response[0])
                supervisor_messages.append({"role": "user", "content": supervisor_next_prompt})
                
                # Add to conversation log
                conversation_log["conversation"].append({
                    "user": "remote",
                    "prompt": supervisor_next_prompt,
                    "output": None,
                })
                
                # Get supervisor's response
                supervisor_response, supervisor_usage = self._get_supervisor_response(supervisor_messages)
                remote_usage += supervisor_usage
                
                # Parse supervisor's response
                supervisor_content = supervisor_response[0]
                
                # Try to extract JSON from supervisor response
                try:
                    supervisor_json = json.loads(supervisor_content)
                except json.JSONDecodeError:
                    supervisor_json = extract_json(supervisor_content)
                
                # Check if supervisor wants to end conversation
                supervisor_decision = supervisor_json.get("decision", "")
                if supervisor_decision == "end_conversation":
                    final_answer = supervisor_json.get("answer", "")
                    conversation_log["generated_final_answer"] = final_answer
                    break
                
                # Add supervisor's response
                supervisor_messages.append({"role": "assistant", "content": supervisor_content})
                conversation_log["conversation"][-1]["output"] = supervisor_content
                
                if self.callback:
                    self.callback("supervisor", supervisor_messages[-1])
                
                # Extract message for worker
                supervisor_message = supervisor_json.get("message", supervisor_content)
                
                # Add to worker messages
                worker_messages.append({"role": "user", "content": supervisor_message})
                
                # Add to conversation log
                conversation_log["conversation"].append({
                    "user": "local",
                    "prompt": supervisor_message,
                    "output": None,
                })
        
        # If we don't have a final answer yet, get one from the worker
        if not final_answer:
            final_answer = self._generate_final_answer(task, worker_messages)
            conversation_log["generated_final_answer"] = final_answer
        
        return final_answer
        
    def _create_supervisor_prompt(self, worker_response: str) -> str:
        """Create the next prompt for the supervisor based on worker's response."""
        return f"""
        ### Worker's Response
        {worker_response}
        
        ### Instructions
        Based on the worker's response above, decide whether to:
        1. Ask another question to guide the worker towards a better answer
        2. End the conversation because the worker has provided sufficient information
        
        If asking another question:
        - Be specific about what additional information you need
        - Ask only one question at a time
        - Refer to the worker's response to build on their thinking
        
        If ending the conversation:
        - Only do this if the worker has fully addressed the task
        
        Provide your decision as a JSON object:
        
        ```json
        {{
            "decision": "ask_followup_question",
            "message": "<your question to the worker>"
        }}
        ```
        
        OR
        
        ```json
        {{
            "decision": "end_conversation",
            "answer": "<final answer to the original task>"
        }}
        ```
        """
    
    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract final answer from response if present."""
        # Check for completion signal in the response
        if "status" in response and "complete" in response:
            try:
                # Try to parse JSON response
                data = json.loads(response)
                if data.get("status") == "complete" and "answer" in data:
                    return data["answer"]
            except:
                # If not valid JSON, check for other completion indicators
                pass
                
        # Check for other completion indicators
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:")
            if len(parts) > 1:
                return parts[1].strip()
                
        return None
    
    def _generate_final_answer(self, task: str, worker_messages: List[Dict[str, str]]) -> str:
        """Generate a final answer if one wasn't provided during conversation."""
        final_answer_prompt = f"""
        Based on our conversation so far, please provide your final answer to the original task:
        
        "{task}"
        
        Please be comprehensive but concise. Format your answer for readability.
        """
        
        # Get final answer from worker
        final_answer_result, _, _ = self.local_client.chat(
            messages=worker_messages + [{"role": "user", "content": final_answer_prompt}]
        )
        
        return final_answer_result[0]
    
    def _save_conversation_log(
        self, 
        conversation_log: Dict[str, Any], 
        logging_id: Optional[str]
    ) -> str:
        """Save the conversation log to a file."""
        if logging_id:
            filename = f"{self.log_dir}/{logging_id}.json"
        else:
            filename = f"{self.log_dir}/minion_conversation_{self._session_timestamp}.json"
            
        try:
            with open(filename, "w") as f:
                json.dump(conversation_log, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation log: {e}")
            
        return filename