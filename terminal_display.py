import tkinter as tk
from tkinter import ttk
import sys
import threading
import queue
import time

# Queue for communication between threads
output_queue = queue.Queue()

class StdoutRedirector:
    """Redirects stdout to our application"""
    def __init__(self, queue):
        self.queue = queue
        self.original_stdout = sys.stdout
    
    def write(self, text):
        # Write to both the queue and the original stdout
        self.queue.put(text)
        self.original_stdout.write(text)
    
    def flush(self):
        self.original_stdout.flush()

def process_output():
    """Process any available output in the queue"""
    try:
        while True:
            # Get output without blocking
            text = output_queue.get_nowait()
            if text:
                # Add text to the display
                text_display.config(state=tk.NORMAL)
                text_display.insert(tk.END, text)
                text_display.see(tk.END)  # Auto-scroll to the end
                text_display.config(state=tk.DISABLED)
    except queue.Empty:
        # No output available, that's okay
        pass
    
    # Schedule this function to run again after 100ms
    root.after(100, process_output)

def demo_output():
    """Generate some demo text to show the functionality"""
    for i in range(10):
        print(f"This is line {i+1} of terminal output")
        time.sleep(0.5)
    print("\nTerminal output complete!")

# Create the main window
root = tk.Tk()
root.title("Terminal Output Display")
root.geometry("600x400")

# Create a frame with padding
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

# Add a label
ttk.Label(frame, text="Terminal Output:").pack(anchor=tk.W, pady=(0, 5))

# Create scrollable text widget to display output
text_display = tk.Text(frame, height=15, width=70, state=tk.DISABLED)
text_display.pack(fill=tk.BOTH, expand=True)

# Add scrollbar
scrollbar = ttk.Scrollbar(text_display, command=text_display.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_display.config(yscrollcommand=scrollbar.set)

# Redirect stdout to our application
sys.stdout = StdoutRedirector(output_queue)

# Schedule the output processor
root.after(100, process_output)

# Start a demo thread to show some output
demo_thread = threading.Thread(target=demo_output, daemon=True)
demo_thread.start()

# Start the main loop
root.mainloop()

# Restore stdout when done
sys.stdout = sys.__stdout__ 