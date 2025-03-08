from minions.clients.ollama import OllamaClient
from minions.minion import Minion
import argparse
import time  # Added for streaming delays
import sys  # Added for flushing output
# from voice import voice


# Add command line arguments for display options
parser = argparse.ArgumentParser(description='Run minions conversation with display options')
parser.add_argument('--full-messages', action='store_true', help='Display full messages without truncation')
parser.add_argument('--no-color', action='store_true', help='Disable colored output')
parser.add_argument('--no-stream', action='store_true', help='Disable streaming output')  # Added streaming option
parser.add_argument('--stream-delay', type=float, default=0.01, help='Delay between characters when streaming (default: 0.01s)')  # Added delay configuration
args = parser.parse_args()

# Configure the clients with appropriate parameters
local_client = OllamaClient(
    model_name="deepseek-r1:1.5b",
    temperature=0.2,  # Lower temperature for more deterministic outputs
)
    
remote_client = OllamaClient(
    model_name="llama3.2:3b",
    temperature=0.1,  # Lower temperature for more structured outputs
)

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)


context = """
You are participating in a two-agent AI collaboration system with these roles:
- LOCAL MODEL : Worker role, model ID "Worker", using deepseek-r1:1.5b
- REMOTE MODEL: Supervisor role, model ID "Supervisor", using llama3.2:3b

COMMUNICATION RULES:
1. Always begin messages to the other agent with their ID (e.g., "@Supervisor: " or "@Worker: ")
2. Keep messages concise (under 100 characters) but informative enough to collaborate
3. Use casual, natural language like humans do, with simple terms and occasional typos
4. For specific requests, use JSON format: {"request":"action","data":"value"}

COLLABORATION PROTOCOL:
1. Listen for messages prefixed with your ID
2. Acknowledge received messages briefly
3. Focus on solving the task through iterative exchanges
4. Share your thinking process and reasoning when relevant
5. When reaching consensus on the task, indicate with: {"status":"complete","answer":"your solution"}

Remember that you're working as a team to solve the given task, and neither agent has complete information alone.
"""

# The task to be performed
task = "Do humans need to be destroyed because their evil nature is always destroying the earth?"

# Execute the minion protocol for up to five communication rounds
output = minion(
    task=task,
    context=[context],
    max_rounds=5
)

# Define ANSI color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Function to apply colors if enabled
def colorize(text, color):
    if args.no_color:
        return text
    return color + text + Colors.END

# Streaming print function that prints character by character with a delay
def stream_print(text, delay=None, end='\n', color=None):
    if delay is None:
        delay = args.stream_delay
    
    if color:
        text = colorize(text, color)
    
    if args.no_stream:
        # If streaming is disabled, use regular print
        print(text, end=end)
        sys.stdout.flush()
        return
    
    # Stream character by character
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    
    # Print the end character (usually a newline)
    print(end=end, flush=True)

# Helper function to clean content by removing Task and Instructions sections
def clean_content(content):
    lines = content.split('\n')
    cleaned_lines = []
    skip_section = False
    
    for line in lines:
        if line.strip().startswith('### Task') or line.strip().startswith('### Instructions'):
            skip_section = True
        elif line.strip().startswith('###'):
            skip_section = False
            cleaned_lines.append(line)
        elif line.strip().startswith('```json'):
            skip_section = True
        elif line.strip().startswith('```') and skip_section:
            skip_section = False
        elif not skip_section:
            cleaned_lines.append(line)
        
    
    return '\n'.join(cleaned_lines)

# Display the conversation in a clearer, interactive format
stream_print("\n" + "=== CONVERSATION HISTORY ===", color=Colors.BOLD + Colors.UNDERLINE)
stream_print("")

# Process the messages to create a more interactive conversation
conversation = []
supervisor_messages = output["supervisor_messages"]
worker_messages = output["worker_messages"]

# Track used questions and answers to prevent duplication
used_questions = set()
used_answers = set()

# Skip the system message in worker_messages
if len(worker_messages) > 0 and worker_messages[0]["role"] == "system":
    worker_messages = worker_messages[1:]

# Process the initial task
if supervisor_messages and len(supervisor_messages) > 0:
    initial_task = supervisor_messages[0]["content"]
    stream_print("🔷 INITIAL TASK:", color=Colors.BOLD + Colors.BLUE)
    stream_print("-" * 80)
    initial_content = clean_content(initial_task)
    if len(initial_content) > 200 and not args.full_messages:
        stream_print(f"{initial_content.strip()[:200]}...\n(Use --full-messages to see complete content)")
    else:
        stream_print(f"{initial_content.strip()}")
    stream_print("-" * 80)
    stream_print("")
    
    # Skip the initial task in supervisor_messages for the rest of the processing
    supervisor_messages = supervisor_messages[1:]

# Create an interleaved conversation
round_num = 1
idx = 0
while idx < max(len(supervisor_messages), len(worker_messages)):
    stream_print(f"🔶 ROUND {round_num}", color=Colors.BOLD + Colors.YELLOW)
    stream_print("-" * 80)
    
    # Track if any content was shown in this round
    round_has_content = False
    
    # Supervisor asks (if there's a question at this point)
    if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "assistant":
        content = clean_content(supervisor_messages[idx]["content"])
        
        # Check if this is a duplicate question
        content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
        if content_key not in used_questions:
            used_questions.add(content_key)
            stream_print("Supervisor (Remote) asks:", color=Colors.BOLD + Colors.BLUE)
            
            if len(content) > 300 and not args.full_messages:
                stream_print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
            else:
                stream_print(f"{content.strip()}")
            stream_print("")
            round_has_content = True
        # voice(content)
    
    # Worker answers (if there's an answer at this point)
    if idx < len(worker_messages) and worker_messages[idx]["role"] == "assistant":
        content = clean_content(worker_messages[idx]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
        if content_key not in used_answers:
            used_answers.add(content_key)
            # Emphasize worker contributions with special formatting and an icon
            stream_print("★ Worker (Local) answers: ★", color=Colors.BOLD + Colors.GREEN + Colors.UNDERLINE)
            
            # For streaming, we need to handle colored replacements differently
            display_content = content
            # Highlight worker's direct thoughts with even more emphasis
            if "⚡ I think" in content or "⚡ In my opinion" in content or "⚡ my analysis" in content:
                # Apply additional formatting to direct worker contributions
                display_content = display_content.replace("⚡ I think", colorize("⚡ I think", Colors.BOLD + Colors.YELLOW))
                display_content = display_content.replace("⚡ In my opinion", colorize("⚡ In my opinion", Colors.BOLD + Colors.YELLOW))
                display_content = display_content.replace("⚡ my analysis", colorize("⚡ my analysis", Colors.BOLD + Colors.YELLOW))
            
            if len(content) > 300 and not args.full_messages:
                stream_print(f"{display_content.strip()[:300]}...\n(Use --full-messages to see complete content)")
            else:
                stream_print(f"{display_content.strip()}")
            stream_print("")
            round_has_content = True
    
    # Worker asks (if there's a next question from worker)
    if idx + 1 < len(worker_messages) and worker_messages[idx + 1]["role"] == "user":
        content = clean_content(worker_messages[idx + 1]["content"])
        
        # Check if this is a duplicate question
        content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
        if content_key not in used_questions:
            used_questions.add(content_key)
            
            # Emphasize worker-initiated questions with even more prominence
            if "★ WORKER QUESTION ★" in content:
                stream_print("★★★ Worker-Initiated Question ★★★", color=Colors.BOLD + Colors.GREEN + Colors.UNDERLINE)
                # Special formatting for worker questions to make them more noticeable
                content = content.replace("★ WORKER QUESTION ★: ", "")
            else:
                # Regular worker question (responding to supervisor)
                stream_print("★ Worker (Local) asks: ★", color=Colors.BOLD + Colors.GREEN + Colors.UNDERLINE)
            
            if len(content) > 300 and not args.full_messages:
                stream_print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
            else:
                stream_print(f"{content.strip()}")
            stream_print("")
            round_has_content = True
    
    # Supervisor answers (if there's a next answer from supervisor)
    if idx + 1 < len(supervisor_messages) and supervisor_messages[idx + 1]["role"] == "user":
        content = clean_content(supervisor_messages[idx + 1]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
        if content_key not in used_answers:
            used_answers.add(content_key)
            stream_print("Supervisor (Remote) answers:", color=Colors.BOLD + Colors.BLUE)
            if len(content) > 300 and not args.full_messages:
                stream_print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
            else:
                stream_print(f"{content.strip()}")
            stream_print("")
            round_has_content = True
    
    # Only increment round number if the round had content
    if round_has_content:
        stream_print("-" * 80)
        stream_print("")
        round_num += 1
        
    idx += 2  # Move to next pair of messages

# Print the final answer with clear separation
stream_print("\n" + "=" * 30 + " FINAL ANSWER " + "=" * 30, color=Colors.BOLD + Colors.RED)
stream_print(output["final_answer"])
stream_print("=" * 80)

# Print a helpful message about command-line options
stream_print("\nTIP: Run with --full-messages to see complete messages without truncation")
stream_print("     Run with --no-color to disable colored output")
stream_print("     Run with --no-stream to disable streaming output")
stream_print("     Run with --stream-delay=X to adjust streaming speed (default: 0.01s)")