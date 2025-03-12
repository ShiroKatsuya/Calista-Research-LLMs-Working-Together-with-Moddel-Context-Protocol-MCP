from minions.clients.ollama import OllamaClient
from minions.minion import Minion
import argparse
# from voice import voice
import sys




def main(task):

    # Add command line arguments for display options
    parser = argparse.ArgumentParser(description='Run minions conversation with display options')
    parser.add_argument('--full-messages', action='store_true', help='Display full messages without truncation')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
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

    COMMUNICATION PROTOCOL:
    1. You are the LOCAL MODEL with the ID "Worker"
    2. You are communicating with a REMOTE MODEL with the ID "Supervisor"
    3. ALWAYS start your messages with "@Supervisor: " when responding
    4. When you receive messages, they will start with "@Worker: "
    5. For specific requests, you can use JSON format: {{"request":"action","data":"value"}}
    6. When you reach a conclusion or have a final answer, indicate with: {{"status":"complete","answer":"your solution"}}


    COLLABORATION PROTOCOL:
    1. Listen for messages prefixed with your ID
    2. Acknowledge received messages briefly
    3. Focus on solving the task through iterative exchanges
    4. Share your thinking process and reasoning when relevant
    5. When reaching consensus on the task, indicate with: {"status":"complete","answer":"your solution"}

    Remember that you're working as a team to solve the given task, and neither agent has complete information alone.
    """

    # The task to be performed
    task = task

    # Execute the minion protocol for up to five communication rounds
    output = minion(
        task=task,
        context=[context],
        max_rounds=1000
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
    print("\n" + colorize("=== CONVERSATION HISTORY ===", Colors.BOLD + Colors.UNDERLINE) + "\n")

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
        print(colorize("🔷 INITIAL TASK:", Colors.BOLD + Colors.BLUE))
        print("-" * 80)
        initial_content = clean_content(initial_task)
        if len(initial_content) > 200 and not args.full_messages:
            print(f"{initial_content.strip()[:200]}...\n(Use --full-messages to see complete content)")
        else:
            print(f"{initial_content.strip()}")
        print("-" * 80 + "\n")
        
        # Skip the initial task in supervisor_messages for the rest of the processing
        supervisor_messages = supervisor_messages[1:]

    # Create an interleaved conversation
    round_num = 1
    idx = 0
    while idx < max(len(supervisor_messages), len(worker_messages)):
        print(colorize(f"🔶 ROUND {round_num}", Colors.BOLD + Colors.YELLOW))
        print("-" * 80)
        
        # Track if any content was shown in this round
        round_has_content = False
        
        # Supervisor asks (if there's a question at this point)
        if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "assistant":
            content = clean_content(supervisor_messages[idx]["content"])
            
            # Check if this is a duplicate question
            content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
            if content_key not in used_questions:
                used_questions.add(content_key)
                print(colorize("Supervisor (Remote) asks:", Colors.BOLD + Colors.BLUE))
                
                if len(content) > 300 and not args.full_messages:
                    print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
                else:
                    print(f"{content.strip()}")
                print()
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
                print(colorize("★ Worker (Local) answers: ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
                
                # Highlight worker's direct thoughts with even more emphasis
                if "⚡ I think" in content or "⚡ In my opinion" in content or "⚡ my analysis" in content:
                    # Apply additional formatting to direct worker contributions
                    content = content.replace("⚡ I think", colorize("⚡ I think", Colors.BOLD + Colors.YELLOW))
                    content = content.replace("⚡ In my opinion", colorize("⚡ In my opinion", Colors.BOLD + Colors.YELLOW))
                    content = content.replace("⚡ my analysis", colorize("⚡ my analysis", Colors.BOLD + Colors.YELLOW))
                
                if len(content) > 300 and not args.full_messages:
                    print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
                else:
                    print(f"{content.strip()}")
                print()
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
                    print(colorize("★★★ Worker-Initiated Question ★★★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
                    # Special formatting for worker questions to make them more noticeable
                    content = content.replace("★ WORKER QUESTION ★: ", "")
                else:
                    # Regular worker question (responding to supervisor)
                    print(colorize("★ Worker (Local) asks: ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
                
                if len(content) > 300 and not args.full_messages:
                    print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
                else:
                    print(f"{content.strip()}")
                print()
                round_has_content = True
        
        # Supervisor answers (if there's a next answer from supervisor)
        if idx + 1 < len(supervisor_messages) and supervisor_messages[idx + 1]["role"] == "user":
            content = clean_content(supervisor_messages[idx + 1]["content"])
            
            # Check if this is a duplicate answer
            content_key = content.strip()[:100]  # Use first 100 chars as key to avoid minor differences
            if content_key not in used_answers:
                used_answers.add(content_key)
                print(colorize("Supervisor (Remote) answers:", Colors.BOLD + Colors.BLUE))
                if len(content) > 300 and not args.full_messages:
                    print(f"{content.strip()[:300]}...\n(Use --full-messages to see complete content)")
                else:
                    print(f"{content.strip()}")
                print()
                round_has_content = True
        
        # Only increment round number if the round had content
        if round_has_content:
            print("-" * 80 + "\n")
            round_num += 1
            
        idx += 2  # Move to next pair of messages

    # Print the final answer with clear separation
    print("\n" + colorize("=" * 30 + " FINAL ANSWER " + "=" * 30, Colors.BOLD + Colors.RED))
    print(output["final_answer"])
    print("=" * 80)

    # Print a helpful message about command-line options
    print("\nTIP: Run with --full-messages to see complete messages without truncation")
    print("     Run with --no-color to disable colored output")