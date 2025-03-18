import tkinter as tk
from tkinter import ttk
import sys
import threading
import queue
import time

# Queue for communication between threads
input_queue = queue.Queue()
output_queue = queue.Queue()

def terminal_input_thread():
    """Thread function to handle terminal input"""
    print("Terminal input ready. Type text and press Enter (type 'exit' to quit):")
    while True:
        user_input = input()
        if user_input.lower() == 'exit':
            input_queue.put(None)  # Signal to exit
            break
        input_queue.put(user_input)

def process_input():
    """Process any available input from the terminal thread"""
    try:
        while True:
            # Get input without blocking
            user_input = input_queue.get_nowait()
            if user_input is None:  # Exit signal
                root.quit()
                return
            # Process the input and update the display with delay
            display_with_delay(user_input)
    except queue.Empty:
        # No input available, schedule next check
        pass
    
    # Schedule this function to run again after 100ms
    root.after(100, process_input)

def display_with_delay(text):
    """Display text character by character with a delay"""
    status_label.config(text="Rendering output...", foreground="orange")
    
    # Clear the current text
    result_label.config(text="")
    root.update()
    
    # Display text character by character
    full_text = f"Input: {text}"
    displayed_text = ""
    
    def add_char(index):
        nonlocal displayed_text
        if index < len(full_text):
            displayed_text += full_text[index]
            result_label.config(text=displayed_text)
            root.after(50, add_char, index + 1)  # 50ms delay between characters
        else:
            # Finished displaying
            status_label.config(text="Terminal ready for input", foreground="green")
    
    # Start the character-by-character display
    add_char(0)

# Create the main window
root = tk.Tk()
root.title("Terminal Input Display")
root.geometry("400x200")

# Create display frame
display_frame = ttk.Frame(root, padding="10")
display_frame.pack(fill=tk.BOTH, expand=True)

# Create label to display result
ttk.Label(display_frame, text="Terminal Input:").pack(anchor=tk.W)
result_label = ttk.Label(display_frame, text="Waiting for input...", font=("Arial", 14))
result_label.pack(pady=10)

# Create status label
status_label = ttk.Label(display_frame, text="Terminal ready for input", foreground="green")
status_label.pack(pady=5)

# Start the terminal input thread
input_thread = threading.Thread(target=terminal_input_thread, daemon=True)
input_thread.start()

# Schedule the first check for input
root.after(100, process_input)

# Start the main loop
root.mainloop()