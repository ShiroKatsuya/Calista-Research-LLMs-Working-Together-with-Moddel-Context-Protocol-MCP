from typing import List, Dict, Any
import json
import re
import os
from datetime import datetime

from minions.clients import OpenAIClient, TogetherClient

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
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        max_rounds=None,
        callback=None,
        log_dir="minion_logs",
    ):
        """Initialize the Minion with local and remote LLM clients.

        Args:
            local_client: Client for the local model (e.g. OllamaClient)
            remote_client: Client for the remote model (e.g. OpenAIClient)
            max_rounds: Maximum number of conversation rounds
            callback: Optional callback function to receive message updates
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.max_rounds = max_rounds
        self.callback = callback
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Track used questions and answers to avoid duplicates
        self.used_questions = set()
        self.used_answers = set()

    def __call__(
        self,
        task: str,
        context: List[str],
        max_rounds=None,
        doc_metadata=None,
        logging_id=None,  # this is the name/id to give to the logging .json file
        is_privacy=False,
    ):
        """Run the minion protocol to answer a task using local and remote models.

        Args:
            task: The task/question to answer
            context: List of context strings
            max_rounds: Override default max_rounds if provided
            doc_metadata: Optional metadata about the documents
            logging_id: Optional identifier for the task, used for named log files

        Returns:
            Dict containing final_answer, conversation histories, and usage statistics
        """

        if max_rounds is None:
            max_rounds = self.max_rounds
            
        # Reset tracking of used questions and answers for this run
        self.used_questions = set()
        self.used_answers = set()

        # Join context sections
        context = "\n\n".join(context)

        # Initialize the log structure
        conversation_log = {
            "task": task,
            "context": context,
            "conversation": [],
            "generated_final_answer": "",
        }

        # Initialize message histories and usage tracking
        supervisor_messages = [
            {
                "role": "user",
                "content": SUPERVISOR_INITIAL_PROMPT.format(task=task),
            }
        ]

        # Add initial supervisor prompt to conversation log
        conversation_log["conversation"].append(
            {
                "user": "remote",
                "prompt": SUPERVISOR_INITIAL_PROMPT.format(task=task),
                "output": None,
            }
        )

        worker_messages = [
            {
                "role": "system",
                "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
            }
        ]

        # if privacy import from minions.utils.pii_extraction
        if is_privacy:
            from minions.utils.pii_extraction import PIIExtractor

            # Extract PII from context
            pii_extractor = PIIExtractor()
            str_context = "\n\n".join(context)
            pii_extracted = pii_extractor.extract_pii(str_context)

            # Extract PII from query
            query_pii_extracted = pii_extractor.extract_pii(task)
            reformat_query_task = REFORMAT_QUERY_PROMPT.format(
                query=task, pii_extracted=str(query_pii_extracted)
            )

            # Clean PII from query
            reformatted_task, usage, done_reason = self.local_client.chat(
                messages=[{"role": "user", "content": reformat_query_task}]
            )
            local_usage += usage
            pii_reformatted_task = reformatted_task[0]

            # Log the reformatted task
            output = f"""**PII Reformated Task:**
            {pii_reformatted_task}
            """

            if self.callback:
                self.callback("worker", output)

            # Initialize message histories
            supervisor_messages = [
                {
                    "role": "user",
                    "content": SUPERVISOR_INITIAL_PROMPT.format(
                        task=pii_reformatted_task
                    ),
                }
            ]
            worker_messages = [
                {
                    "role": "system",
                    "content": WORKER_SYSTEM_PROMPT.format(context=context, task=task),
                }
            ]
        else:
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

        if max_rounds is None:
            max_rounds = self.max_rounds

        # Initial supervisor call to get first question
        if self.callback:
            self.callback("supervisor", None, is_final=False)

        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            supervisor_response, supervisor_usage = self.remote_client.chat(
                messages=supervisor_messages, response_format={"type": "json_object"}
            )
        else:
            supervisor_response, supervisor_usage, _ = self.remote_client.chat(
                messages=supervisor_messages
            )

        remote_usage = Usage()
        local_usage = Usage()

        remote_usage += supervisor_usage
        supervisor_messages.append(
            {"role": "assistant", "content": supervisor_response[0]}
        )

        # Update the last conversation entry with the output
        conversation_log["conversation"][-1]["output"] = supervisor_response[0]

        if self.callback:
            self.callback("supervisor", supervisor_messages[-1])

        # Extract first question for worker
        if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
            try:
                if supervisor_response and len(supervisor_response) > 0:
                    supervisor_json = json.loads(supervisor_response[0])
                else:
                    supervisor_json = {
                        "decision": "provide_final_answer",
                        "answer": "Empty supervisor response received",
                        "message": "No supervisor response available"
                    }
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try the extract_json function
                if supervisor_response and len(supervisor_response) > 0:
                    supervisor_json = _extract_json(supervisor_response[0])
                else:
                    supervisor_json = {
                        "decision": "provide_final_answer",
                        "answer": "Empty supervisor response received",
                        "message": "No supervisor response available"
                    }
        else:
            if supervisor_response and len(supervisor_response) > 0:
                supervisor_json = _extract_json(supervisor_response[0])
            else:
                supervisor_json = {
                    "decision": "provide_final_answer",
                    "answer": "Empty supervisor response received",
                    "message": "No supervisor response available"
                }
        
        # Check if this is a unique question
        question_key = supervisor_json["message"].strip()[:100]  # Use first 100 chars as key
        if question_key in self.used_questions:
            # Question already used, generate a more specific question
            supervisor_json["message"] += " Please be more specific and provide a unique perspective on this topic that we haven't explored before."
        
        # Add to used questions
        self.used_questions.add(question_key)
                
        worker_messages.append({"role": "user", "content": supervisor_json["message"]})

        # Add worker prompt to conversation log
        conversation_log["conversation"].append(
            {"user": "local", "prompt": supervisor_json["message"], "output": None}
        )

        final_answer = None
        for round in range(max_rounds):
            # Get worker's response
            if self.callback:
                self.callback("worker", None, is_final=False)

            worker_response, worker_usage, done_reason = self.local_client.chat(
                messages=worker_messages
            )

            local_usage += worker_usage
            
            # Check for uniqueness of worker's response
            response_key = worker_response[0].strip()[:100]  # Use first 100 chars as key
            
            if response_key in self.used_answers:
                # Worker provided a duplicate answer, request a more unique response
                worker_uniqueness_prompt = """Your previous response was too similar to one you've already provided. 
                Please provide a more unique and detailed perspective on the topic that adds new information.
                Be specific and try to approach the question from a different angle. 
                Avoid repeating similar points, phrases, or structures from your previous responses."""
                
                worker_messages.append({"role": "user", "content": worker_uniqueness_prompt})
                
                worker_response, worker_usage, done_reason = self.local_client.chat(
                    messages=worker_messages
                )
                
                # Remove the uniqueness prompt from messages to avoid confusion in future exchanges
                worker_messages.pop()
                
                local_usage += worker_usage
            
            # Add to used answers
            self.used_answers.add(response_key)

            if is_privacy:
                if self.callback:
                    output = f"""**_My output (pre-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)

                worker_privacy_shield_prompt = WORKER_PRIVACY_SHIELD_PROMPT.format(
                    output=worker_response[0],
                    pii_extracted=str(pii_extracted),
                )
                worker_response, worker_usage, done_reason = self.local_client.chat(
                    messages=[{"role": "user", "content": worker_privacy_shield_prompt}]
                )
                local_usage += worker_usage

                worker_messages.append(
                    {"role": "assistant", "content": worker_response[0]}
                )
                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = worker_response[0]

                if self.callback:
                    output = f"""**_My output (post-privacy shield):_**

                    {worker_response[0]}
                    """
                    self.callback("worker", output)
            else:
                # Emphasize worker messages by adding a prefix to distinguish them
                emphasized_response = worker_response[0]
                # Check if the message already starts with "@Supervisor:"
                if not emphasized_response.startswith("@Supervisor:"):
                    emphasized_response = "@Supervisor: " + emphasized_response
                else:
                    # If it already has the prefix, make sure we're not adding it twice
                    emphasized_response = emphasized_response
                
                # Add emphasis for worker's direct thoughts/contributions
                if "I think" in emphasized_response or "In my opinion" in emphasized_response or "my analysis" in emphasized_response:
                    # Highlight worker's direct thoughts with bold markers
                    emphasized_response = emphasized_response.replace("I think", "⚡ I think")
                    emphasized_response = emphasized_response.replace("In my opinion", "⚡ In my opinion")
                    emphasized_response = emphasized_response.replace("my analysis", "⚡ my analysis")
                
                # Additional worker-specific phrases to highlight for more emphasis
                worker_phrases = [
                    "From my perspective", 
                    "My understanding is", 
                    "I believe", 
                    "I'd suggest", 
                    "I propose",
                    "my view",
                    "my interpretation",
                    "I'm thinking",
                    "I disagree",
                    "I agree"
                ]
                
                for phrase in worker_phrases:
                    if phrase in emphasized_response:
                        emphasized_response = emphasized_response.replace(phrase, f"⚡ {phrase}")
                
                worker_messages.append(
                    {"role": "assistant", "content": emphasized_response}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = emphasized_response

                if self.callback:
                    self.callback("worker", worker_messages[-1])

            # NEW: Let the worker model initiate a question back to the supervisor
            # without requiring the supervisor to specifically ask for more information
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
            
            worker_question_response, worker_follow_up_usage, _ = self.local_client.chat(
                messages=worker_messages + [{"role": "user", "content": worker_follow_up_prompt}]
            )
            
            local_usage += worker_follow_up_usage
            
            # Check if worker has a follow-up question and it's not a duplicate
            if worker_question_response and worker_question_response[0].strip() != "No questions at this time.":
                question_key = worker_question_response[0].strip()[:100]
                
                # Check if this is a unique question
                if question_key not in self.used_questions:
                    self.used_questions.add(question_key)
                    
                    # Emphasize that this is a worker-initiated question
                    emphasized_question = "★ WORKER QUESTION ★: " + worker_question_response[0]
                    
                    # Add worker's question to conversation log
                    conversation_log["conversation"].append(
                        {"user": "local", "prompt": emphasized_question, "output": None}
                    )
                    
                    # Send worker's question to supervisor
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
                    
                    supervisor_messages.append({"role": "user", "content": supervisor_reply_prompt})
                    
                    # Get supervisor's response
                    if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                        supervisor_reply, supervisor_reply_usage = self.remote_client.chat(
                            messages=supervisor_messages, response_format={"type": "json_object"}
                        )
                    else:
                        supervisor_reply, supervisor_reply_usage, _ = self.remote_client.chat(
                            messages=supervisor_messages
                        )
                    
                    remote_usage += supervisor_reply_usage
                    supervisor_messages.append({"role": "assistant", "content": supervisor_reply[0]})
                    
                    # Update conversation log with supervisor's answer
                    conversation_log["conversation"][-1]["output"] = supervisor_reply[0]
                    
                    # Extract supervisor's answer for worker and add to used answers
                    if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                        try:
                            supervisor_reply_json = json.loads(supervisor_reply[0])
                        except json.JSONDecodeError:
                            supervisor_reply_json = _extract_json(supervisor_reply[0])
                    else:
                        supervisor_reply_json = _extract_json(supervisor_reply[0])
                    
                    # Add the answer to used answers to avoid duplication
                    answer_key = supervisor_reply_json["message"].strip()[:100]
                    self.used_answers.add(answer_key)
                    
                    # Add supervisor's answer to worker's messages
                    worker_messages.append({"role": "user", "content": supervisor_reply_json["message"]})
                    
                    # Add supervisor's answer to conversation log for next worker response
                    conversation_log["conversation"].append(
                        {"user": "local", "prompt": supervisor_reply_json["message"], "output": None}
                    )

            # Format prompt based on whether this is the final round
            if round == max_rounds - 1:
                supervisor_prompt = SUPERVISOR_FINAL_PROMPT.format(
                    response=worker_response[0]
                )

                # Add supervisor final prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )
            else:
                # First step: Think through the synthesis
                cot_prompt = REMOTE_SYNTHESIS_COT.format(response=worker_response[0])

                # Add supervisor COT prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": cot_prompt, "output": None}
                )

                supervisor_messages.append({"role": "user", "content": cot_prompt})

                step_by_step_response, usage, _ = self.remote_client.chat(
                    supervisor_messages
                )

                remote_usage += usage

                supervisor_messages.append(
                    {"role": "assistant", "content": step_by_step_response[0]}
                )

                # Update the last conversation entry with the output
                conversation_log["conversation"][-1]["output"] = step_by_step_response[
                    0
                ]

                # Second step: Get structured output
                supervisor_prompt = REMOTE_SYNTHESIS_FINAL.format(
                    response=step_by_step_response[0]
                )

                # Add supervisor synthesis prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "remote", "prompt": supervisor_prompt, "output": None}
                )

            supervisor_messages.append({"role": "user", "content": supervisor_prompt})

            if self.callback:
                self.callback("supervisor", None, is_final=False)

            # Get supervisor's response
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                supervisor_response, supervisor_usage = self.remote_client.chat(
                    messages=supervisor_messages,
                    response_format={"type": "json_object"},
                )
            else:
                supervisor_response, supervisor_usage, _ = self.remote_client.chat(
                    messages=supervisor_messages
                )

            remote_usage += supervisor_usage
            supervisor_messages.append(
                {"role": "assistant", "content": supervisor_response[0]}
            )
            if self.callback:
                self.callback("supervisor", supervisor_messages[-1])

            conversation_log["conversation"][-1]["output"] = supervisor_response[0]

            # Parse supervisor's decision
            if isinstance(self.remote_client, (OpenAIClient, TogetherClient)):
                try:
                    if supervisor_response and len(supervisor_response) > 0:
                        supervisor_json = json.loads(supervisor_response[0])
                    else:
                        supervisor_json = {
                            "decision": "provide_final_answer",
                            "answer": "Empty supervisor response received",
                            "message": "No supervisor response available"
                        }
                except json.JSONDecodeError:
                    # If direct JSON parsing fails, try the extract_json function
                    if supervisor_response and len(supervisor_response) > 0:
                        supervisor_json = _extract_json(supervisor_response[0])
                    else:
                        supervisor_json = {
                            "decision": "provide_final_answer",
                            "answer": "Empty supervisor response received",
                            "message": "No supervisor response available"
                        }
            else:
                if supervisor_response and len(supervisor_response) > 0:
                    supervisor_json = _extract_json(supervisor_response[0])
                else:
                    supervisor_json = {
                        "decision": "provide_final_answer",
                        "answer": "Empty supervisor response received",
                        "message": "No supervisor response available"
                    }

            # Check if decision key exists in supervisor_json
            if "decision" in supervisor_json and supervisor_json["decision"] == "provide_final_answer":
                final_answer = supervisor_json.get("answer", "No answer provided")
                conversation_log["generated_final_answer"] = final_answer
                break
            else:
                next_question = supervisor_json["message"]
                
                # Check if this question is unique
                question_key = next_question.strip()[:100]
                if question_key in self.used_questions:
                    # Question already used, modify it to be more specific
                    if "@Worker:" in next_question:
                        parts = next_question.split("@Worker:", 1)
                        next_question = "@Worker: " + parts[1].strip() + " Please provide a unique perspective or specific details that haven't been discussed in previous rounds."
                    else:
                        next_question += " Please provide a unique perspective or specific details that haven't been covered in previous rounds."
                
                # Add to used questions
                self.used_questions.add(question_key)
                
                worker_messages.append({"role": "user", "content": next_question})

                # Add next worker prompt to conversation log
                conversation_log["conversation"].append(
                    {"user": "local", "prompt": next_question, "output": None}
                )

        if final_answer is None:
            final_answer = "No answer found."
            conversation_log["generated_final_answer"] = final_answer

        # Log the final result
        if logging_id:
            # use provided logging_id
            log_filename = f"{logging_id}_minion.json"
        else:
            # fall back to timestamp + task abbrev
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_task = re.sub(r"[^a-zA-Z0-9]", "_", task[:15])
            log_filename = f"{timestamp}_{safe_task}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)

        return {
            "final_answer": final_answer,
            "supervisor_messages": supervisor_messages,
            "worker_messages": worker_messages,
            "remote_usage": remote_usage,
            "local_usage": local_usage,
            "log_file": log_path,
        }