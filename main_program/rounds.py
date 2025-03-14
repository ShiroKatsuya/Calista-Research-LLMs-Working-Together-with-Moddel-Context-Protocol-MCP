from typing import Dict, Any, Set, List
from main_program.website_simulations import colorize,Colors,process_supervisor_answer,process_supervisor_question,process_worker_answer,process_worker_question,clean_content


def display_conversation(output: Dict[str, Any], args) -> None:
    colorize.no_color = args.no_color
    print("\n" + colorize("=== CONVERSATION HISTORY ===", Colors.BOLD + Colors.UNDERLINE) + "\n")
    used_questions: Set[str] = set()
    used_answers: Set[str] = set()
    supervisor_messages = output["supervisor_messages"]
    worker_messages = output["worker_messages"]
    if worker_messages and worker_messages[0]["role"] == "system":
        worker_messages = worker_messages[1:]
    if supervisor_messages:
        display_initial_task(supervisor_messages[0]["content"], args.full_messages)
        supervisor_messages = supervisor_messages[1:]
    round_num = 1
    idx = 0
    while idx < max(len(supervisor_messages), len(worker_messages)):
        print(colorize(f"🔶 ROUND {round_num}", Colors.BOLD + Colors.YELLOW))
        print("-" * 80)
        round_has_content = False
        round_has_content |= process_supervisor_question(supervisor_messages, idx, used_questions, args.full_messages)
        round_has_content |= process_worker_answer(worker_messages, idx, used_answers, args.full_messages)
        round_has_content |= process_worker_question(worker_messages, idx + 1, used_questions, args.full_messages)
        round_has_content |= process_supervisor_answer(supervisor_messages, idx + 1, used_answers, args.full_messages)
        if round_has_content:
            print("-" * 80 + "\n")
            round_num += 1
        idx += 2  
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
