import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import threading
import queue
import io
import time
import argparse
import re
from main import main, Colors, colorize

# Queue for storing output from the main program
output_queue = queue.Queue()

class StdoutRedirector(io.StringIO):
    """Redirects stdout to the queue"""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def write(self, string):
        self.queue.put(string)
        
    def flush(self):
        pass

class AnsiColorizer:
    """Handles ANSI color code conversion to Tkinter text tags"""
    # ANSI color codes and their corresponding Tkinter tag configurations
    ANSI_COLOR_MAP = {
        Colors.BLUE: {'foreground': '#3498db'},
        Colors.GREEN: {'foreground': '#2ecc71'},
        Colors.YELLOW: {'foreground': '#f1c40f'},
        Colors.RED: {'foreground': '#e74c3c'},
        Colors.BOLD: {'font': ('Consolas', 10, 'bold')},
        Colors.UNDERLINE: {'underline': True},
    }
    
    def __init__(self, text_widget):
        """Initialize with a text widget to apply colors to"""
        self.text_widget = text_widget
        self._create_tags()
        
    def _create_tags(self):
        """Create tags for each ANSI color code"""
        for ansi_code, config in self.ANSI_COLOR_MAP.items():
            tag_name = self._ansi_to_tag_name(ansi_code)
            self.text_widget.tag_configure(tag_name, **config)
            
    def _ansi_to_tag_name(self, ansi_code):
        """Convert ANSI code to a valid tag name"""
        # Replace special characters with underscores
        return f"ansi_{hash(ansi_code) % 10000}"
    
    def apply_ansi_colors(self, text):
        """Parse text with ANSI codes and insert into text widget with appropriate tags"""
        current_pos = "end"
        remaining_text = text
        
        # Find all ANSI codes in the text
        pattern = re.compile(r'(\033\[[0-9;]*m)')
        
        # Keep track of active tags
        active_tags = []
        
        # Split the text by ANSI codes
        parts = pattern.split(remaining_text)
        
        for part in parts:
            if part == Colors.END:
                # Reset all tags
                active_tags = []
            elif part.startswith('\033['):
                # ANSI code - add to active tags if we recognize it
                for ansi_code in self.ANSI_COLOR_MAP:
                    if part == ansi_code:
                        active_tags.append(self._ansi_to_tag_name(ansi_code))
            else:
                # Regular text - insert with active tags
                if part:
                    self.text_widget.insert("end", part, tuple(active_tags))
                    
        return "end"

class MinionTerminal:
    def __init__(self, root):
        self.root = root
        self.root.title("Minions Terminal - Chat & Run Tasks")
        self.root.geometry("800x600")
        self.root.configure(bg="#1F2C34")
        
        # Set app icon if available
        try:
            self.root.iconbitmap("assets/app_icon.ico")
        except:
            pass  # Ignore if icon not available
        
        # Configure colors to match the app style
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Segoe UI', 12))
        self.style.configure('TLabel', font=('Segoe UI', 12), background="#1F2C34", foreground="white")
        self.style.configure('TFrame', background="#1F2C34")
        
        # Loading indicator
        self.loading = False
        self.loading_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self.loading_index = 0
        
        self.setup_ui()
        
        # Initialize ANSI colorizer
        self.colorizer = AnsiColorizer(self.output_text)
        
        # Start processing the output queue
        self.process_output()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for options
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Task input field
        ttk.Label(top_frame, text="Task:").pack(side=tk.LEFT, padx=(0, 5))
        self.task_entry = ttk.Entry(top_frame, width=50)
        self.task_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.task_entry.insert(0, "Tell me about the Fermi Paradox")
        
        # Bind Enter key to start conversation
        self.task_entry.bind("<Return>", lambda e: self.start_minion_conversation())
        
        # Full messages checkbox
        self.full_messages_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame, 
            text="Full Messages", 
            variable=self.full_messages_var, 
            style='TCheckbutton'
        ).pack(side=tk.LEFT, padx=(10, 5))
        
        # No color checkbox
        self.no_color_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_frame, 
            text="No Color", 
            variable=self.no_color_var, 
            style='TCheckbutton'
        ).pack(side=tk.LEFT, padx=(10, 5))
        
        # Start button
        ttk.Button(
            top_frame, 
            text="Start Conversation", 
            command=self.start_minion_conversation
        ).pack(side=tk.RIGHT, padx=5)
        
        # Clear button
        ttk.Button(
            top_frame, 
            text="Clear", 
            command=self.clear_output
        ).pack(side=tk.RIGHT, padx=5)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD,
            font=("Consolas", 10),
            background="#121B22",
            foreground="white",
            insertbackground="white",
            height=20
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Message entry area
        entry_frame = ttk.Frame(main_frame)
        entry_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add attachment and camera buttons
        attachment_btn = tk.Label(
            entry_frame,
            text="📎",
            font=("Segoe UI", 16),
            fg="#8696A0",
            bg="#1F2C34"
        )
        attachment_btn.pack(side=tk.LEFT, padx=5)
        
        camera_btn = tk.Label(
            entry_frame,
            text="📷",
            font=("Segoe UI", 16),
            fg="#8696A0",
            bg="#1F2C34"
        )
        camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Add hover effect to buttons
        for btn in [attachment_btn, camera_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg="#FFFFFF"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg="#8696A0"))
        
        # Text input field
        self.text_entry = tk.Text(
            entry_frame, 
            wrap=tk.WORD, 
            bg="#1F2C34", 
            fg="white", 
            insertbackground="white",
            height=3, 
            font=("Segoe UI", 11), 
            relief=tk.FLAT, 
            padx=10, 
            pady=10
        )
        self.text_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        
        # Placeholder text
        self.text_entry.insert("1.0", "Type a message...")
        self.text_entry.config(fg="#8696A0")
        
        # Clear placeholder on focus
        def on_text_focus_in(event):
            if self.text_entry.get("1.0", "end-1c") == "Type a message...":
                self.text_entry.delete("1.0", tk.END)
                self.text_entry.config(fg="white")
                
        def on_text_focus_out(event):
            if not self.text_entry.get("1.0", "end-1c").strip():
                self.text_entry.delete("1.0", tk.END)
                self.text_entry.insert("1.0", "Type a message...")
                self.text_entry.config(fg="#8696A0")
                
        self.text_entry.bind("<FocusIn>", on_text_focus_in)
        self.text_entry.bind("<FocusOut>", on_text_focus_out)
        
        # Bind Enter key to send message
        self.text_entry.bind("<Return>", lambda e: self.send_message())
        self.text_entry.bind("<Shift-Return>", lambda e: self.text_entry.insert(tk.END, "\n"))
        
        # Bind Escape key to clear input
        self.text_entry.bind("<Escape>", lambda e: self.text_entry.delete("1.0", tk.END))
        
        # Send button
        send_btn = tk.Button(
            entry_frame,
            text="→",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#00BFA5",
            relief=tk.FLAT,
            width=3,
            command=self.send_message
        )
        send_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Hover effect for send button
        send_btn.bind("<Enter>", lambda e: send_btn.config(bg="#00AD96"))
        send_btn.bind("<Leave>", lambda e: send_btn.config(bg="#00BFA5"))
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(fill=tk.X, pady=(10, 0), anchor=tk.W)
    
    def start_minion_conversation(self):
        """Start the minion conversation in a separate thread"""
        self.clear_output()
        self.status_label.config(text="Conversation started...")
        
        # Start loading animation
        self.loading = True
        self.update_loading_animation()
        
        # Get parameters
        task = self.task_entry.get().strip()
        full_messages = self.full_messages_var.get()
        no_color = self.no_color_var.get()
        
        # Create args object to mimic command line arguments
        class Args:
            pass
        
        args = Args()
        args.full_messages = full_messages
        args.no_color = no_color
        
        # Create a thread to run the minion conversation
        self.conversation_thread = threading.Thread(
            target=self.run_minion,
            args=(task, args),
            daemon=True
        )
        self.conversation_thread.start()
    
    def update_loading_animation(self):
        """Update the loading animation in the status bar"""
        if self.loading:
            loading_char = self.loading_chars[self.loading_index]
            self.status_label.config(text=f"{loading_char} Processing... {loading_char}")
            self.loading_index = (self.loading_index + 1) % len(self.loading_chars)
            self.root.after(100, self.update_loading_animation)
    
    def run_minion(self, task, args):
        """Run the minion with the given parameters"""
        # Redirect stdout to capture the output
        original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(output_queue)
        
        try:
            # Set the colorize attributes for no_color option
            colorize.no_color = args.no_color
            
            # Ensure the main function knows about full_messages and no_color
            # This is needed because in main.py, these are typically parsed from sys.argv
            sys.argv = [sys.argv[0]]
            if args.full_messages:
                sys.argv.append("--full-messages")
            if args.no_color:
                sys.argv.append("--no-color")
            
            # Run the main function with the given parameters
            main(task)  # Directly call main with the task parameter
            
        except Exception as e:
            output_queue.put(f"\n{Colors.RED}Error: {str(e)}{Colors.END}\n")
            self.status_label.config(text=f"Error: {str(e)}")
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            output_queue.put("\nConversation completed.")
            self.status_label.config(text="Conversation completed")
            # Stop loading animation
            self.loading = False
    
    def process_output(self):
        """Process any available output from the queue"""
        try:
            while True:
                # Get output without blocking
                output = output_queue.get_nowait()
                # Apply ANSI colors to the output
                self.colorizer.apply_ansi_colors(output)
                self.output_text.see(tk.END)  # Auto-scroll to the end
        except queue.Empty:
            # No output available, schedule next check
            pass
        
        # Schedule this function to run again after 100ms
        self.root.after(100, self.process_output)
    
    def send_message(self):
        """Handle sending a message"""
        message = self.text_entry.get("1.0", "end-1c").strip()
        if message and message != "Type a message...":
            # Clear the input field
            self.text_entry.delete("1.0", tk.END)
            
            # Set focus back to the text entry
            self.text_entry.focus_set()
            
            # Add the message to task entry and start conversation
            self.task_entry.delete(0, tk.END)
            self.task_entry.insert(0, message)
            
            # Automatically start the conversation
            self.start_minion_conversation()
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete("1.0", tk.END)
        self.status_label.config(text="Output cleared")

if __name__ == "__main__":
    # Create and run the application
    root = tk.Tk()
    app = MinionTerminal(root)
    root.mainloop() 