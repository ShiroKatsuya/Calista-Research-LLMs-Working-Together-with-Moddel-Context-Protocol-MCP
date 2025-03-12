from minions.clients.ollama import OllamaClient
from minions.minion import Minion
import argparse
# from voice import voice
import sys
from typing import Dict, Any, Set
from websitecall import contexts



def main(task: str = None) -> None:
    """
    Main function to run the minion conversation system with a given task.
    
    Args:
        task: The task/question to be answered by the minion system
    """
    # Add command line arguments for display options
    parser = argparse.ArgumentParser(description='Run minions conversation with display options')
    parser.add_argument('task', nargs='?', default=task or "Tell me about the Fermi Paradox and why we haven't found alien life yet.", 
                        help='The task/question to be answered by the minion system')
    parser.add_argument('--full-messages', action='store_true', help='Display full messages without truncation')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    # If running from command line, parse args; otherwise, use provided args
    if 'pytest' not in sys.modules and len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # For testing or when called programmatically
        class Args:
            pass
        args = Args()
        args.task = task or "Tell me about the Fermi Paradox and why we haven't found alien life yet."
        args.full_messages = True  # Default to full messages
        args.no_color = getattr(parser.parse_args([]), 'no_color', False)

    # Use the task from args
    task = args.task

    # Configure the clients with appropriate parameters
    local_client = OllamaClient(
        model_name="llama3.2:1b",
        temperature=0.2,  # Lower temperature for more deterministic outputs
    )
        
    remote_client = OllamaClient(
        model_name="llama3.2:3b",
        temperature=0.1,  # Lower temperature for more structured outputs
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
    """Apply color formatting to text if enabled.
    
    Args:
        text: The text to colorize
        color: The ANSI color code to apply
        
    Returns:
        Colorized text or plain text if colors are disabled
    """
    if getattr(colorize, 'no_color', False):
        return text
    return color + text + Colors.END


def clean_content(content: str) -> str:
    # """Remove Task and Instructions sections from message content.
    
    # Args:
    #     content: The raw message content
        
    # Returns:
    #     Cleaned content with sections removed
    # """
    # lines = content.split('\n')
    # cleaned_lines = []
    # skip_section = False
    
    # for line in lines:
    #     if line.strip().startswith('### Task') or line.strip().startswith('### Instructions'):
    #         skip_section = True
    #     elif line.strip().startswith('###'):
    #         skip_section = False
    #         cleaned_lines.append(line)
    #     elif line.strip().startswith('```json'):
    #         skip_section = True
    #     elif line.strip().startswith('```') and skip_section:
    #         skip_section = False
    #     elif not skip_section:
    #         cleaned_lines.append(line)
            
    # return '\n'.join(cleaned_lines)

    return content


def truncate_content(content: str, max_length: int, full_messages: bool) -> str:
    """Truncate content if needed and add ellipsis.
    
    Args:
        content: Content to potentially truncate
        max_length: Maximum length before truncation
        full_messages: Flag to show full messages
        
    Returns:
        Truncated or full content string
    """
    if full_messages:
        return content.strip()
    if len(content) > max_length:
        return f"{content.strip()[:max_length]}...\n(Use --full-messages to see complete content)"
    return content.strip()


def display_conversation(output: Dict[str, Any], args) -> None:
    """Display the conversation in a clearer, interactive format.
    
    Args:
        output: The output dictionary from the minion call
        args: Command-line arguments
    """
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
    """Display the initial task given to the minion system.
    
    Args:
        task_content: The content of the initial task
        full_messages: Flag to show full messages
    """
    print(colorize("🔷 INITIAL TASK:", Colors.BOLD + Colors.BLUE))
    print("-" * 80)
    initial_content = clean_content(task_content)
    print(truncate_content(initial_content, 1000, full_messages))
    print("-" * 80 + "\n")


def process_supervisor_question(supervisor_messages, idx, used_questions, full_messages):
    """Process and display a supervisor question if it exists and is unique.
    
    Returns:
        Boolean indicating if content was shown
    """
    if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "assistant":
        content = clean_content(supervisor_messages[idx]["content"])
        
        # Check if this is a duplicate question
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_questions:
            used_questions.add(content_key)
            print(colorize("Supervisor (Remote) asks:", Colors.BOLD + Colors.BLUE))
            print(truncate_content(content, 2000, full_messages))
            print()
            return True
    return False


def process_worker_answer(worker_messages, idx, used_answers, full_messages):
    """Process and display a worker answer if it exists and is unique.
    
    Returns:
        Boolean indicating if content was shown
    """
    if idx < len(worker_messages) and worker_messages[idx]["role"] == "assistant":
        content = clean_content(worker_messages[idx]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_answers:
            used_answers.add(content_key)
            print(colorize("★ Worker (Local) answers: ★", Colors.BOLD + Colors.GREEN + Colors.UNDERLINE))
            
            # Highlight worker's direct thoughts with emphasis
            highlight_phrases = ["⚡ I think", "⚡ In my opinion", "⚡ my analysis"]
            for phrase in highlight_phrases:
                base_phrase = phrase.replace("⚡ ", "")
                content = content.replace(base_phrase, colorize(phrase, Colors.BOLD + Colors.YELLOW))
            
            print(truncate_content(content, 2000, full_messages))
            print()
            return True
    return False


def process_worker_question(worker_messages, idx, used_questions, full_messages):
    """Process and display a worker question if it exists and is unique.
    
    Returns:
        Boolean indicating if content was shown
    """
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
            
            print(truncate_content(content, 2000, full_messages))
            print()
            return True
    return False


def process_supervisor_answer(supervisor_messages, idx, used_answers, full_messages):
    """Process and display a supervisor answer if it exists and is unique.
    
    Returns:
        Boolean indicating if content was shown
    """
    if idx < len(supervisor_messages) and supervisor_messages[idx]["role"] == "user":
        content = clean_content(supervisor_messages[idx]["content"])
        
        # Check if this is a duplicate answer
        content_key = content.strip()[:100]  # Use first 100 chars as key
        if content_key not in used_answers:
            used_answers.add(content_key)
            print(colorize("Supervisor (Remote) answers:", Colors.BOLD + Colors.BLUE))
            print(truncate_content(content, 2000, full_messages))
            print()
            return True
    return False


if __name__ == "__main__":
    # Run the main function (argument parsing happens inside main)
    main()