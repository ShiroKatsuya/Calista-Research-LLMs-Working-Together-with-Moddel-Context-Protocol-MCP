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
    # Enhanced color map with richer colors
    ANSI_COLOR_MAP = {
        Colors.BLUE: {'foreground': '#3498db'},
        Colors.GREEN: {'foreground': '#2ecc71'},
        Colors.YELLOW: {'foreground': '#f1c40f'},
        Colors.RED: {'foreground': '#e74c3c'},
        Colors.BOLD: {'font': ('Consolas', 10, 'bold')},
        Colors.UNDERLINE: {'underline': True},
        # Additional styling options
        'cyan': {'foreground': '#00bcd4'},
        'magenta': {'foreground': '#e91e63'},
        'orange': {'foreground': '#ff9800'},
        'purple': {'foreground': '#9c27b0'},
        'gray': {'foreground': '#9e9e9e'},
        'highlight': {'background': '#2c3e50', 'foreground': '#ecf0f1'},
        # Enhanced thinking messages styling
        'supervisor_thinking': {'foreground': '#8e44ad', 'font': ('Consolas', 10, 'bold')},
        'worker_thinking': {'foreground': '#d35400', 'font': ('Consolas', 10, 'bold')},
    }
    
    def __init__(self, text_widget):
        """Initialize with a text widget to apply colors to"""
        self.text_widget = text_widget
        self._create_tags()
        
    def _create_tags(self):
        """Create text tags for each ANSI color"""
        for name, config in self.ANSI_COLOR_MAP.items():
            self.text_widget.tag_configure(self._ansi_to_tag_name(name), **config)
            
    def _ansi_to_tag_name(self, ansi_code):
        """Convert ANSI code to tag name"""
        return f"ansi_{ansi_code}"
    
    def apply_ansi_colors(self, text):
        """Apply ANSI colors to the text widget"""
        # Special handling for thinking messages with ANSI codes
        supervisor_pattern = r'\[1m\[94m\s*Supervisor \(Remote\) is thinking\.\.\.'
        worker_pattern = r'\[1m\[92m\[4m\s*★ Worker \(Local\) is thinking\.\.\. ★'
        
        # Check for special messages
        if re.search(supervisor_pattern, text):
            clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
            self.text_widget.insert(tk.END, clean_text, "supervisor_thinking")
            return
        elif re.search(worker_pattern, text):
            clean_text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
            self.text_widget.insert(tk.END, clean_text, "worker_thinking")
            return
            
        # Original ANSI processing for other text
        ansi_pattern = re.compile(r'\x1b\[([\d;]*)m')
        
        # Start position for inserting text
        start_pos = self.text_widget.index(tk.END)
        self.text_widget.insert(tk.END, ansi_pattern.sub('', text))
        
        # Find all matches and apply tags
        for match in ansi_pattern.finditer(text):
            code = match.group(1)
            
            # Determine the tag to apply based on the code
            if code == '0':  # Reset
                continue
            elif code == '1':  # Bold
                tag = self._ansi_to_tag_name(Colors.BOLD)
            elif code == '4':  # Underline
                tag = self._ansi_to_tag_name(Colors.UNDERLINE)
            elif code == '34':  # Blue
                tag = self._ansi_to_tag_name(Colors.BLUE)
            elif code == '32':  # Green
                tag = self._ansi_to_tag_name(Colors.GREEN)
            elif code == '33':  # Yellow
                tag = self._ansi_to_tag_name(Colors.YELLOW)
            elif code == '31':  # Red
                tag = self._ansi_to_tag_name(Colors.RED)
            elif code == '36':  # Cyan
                tag = self._ansi_to_tag_name('cyan')
            elif code == '35':  # Magenta
                tag = self._ansi_to_tag_name('magenta')
            else:
                continue
                
            # Apply tag to the appropriate text segment
            # (This is simplified, would need more logic for real implementation)

# Theme constants for consistent styling
class TerminalTheme:
    BG_COLOR = "#1E1E1E"
    PANEL_COLOR = "#252526"
    TEXT_COLOR = "#E9E9E9"
    ACCENT_COLOR = "#007ACC"
    HIGHLIGHT_COLOR = "#264F78"
    ERROR_COLOR = "#F44747"
    SUCCESS_COLOR = "#6A9955"
    WARNING_COLOR = "#CCA700"
    INPUT_BG = "#3C3C3C"
    BUTTON_BG = "#0E639C"
    BUTTON_HOVER = "#1177BB"
    
    # Enhanced colors for thinking messages
    SUPERVISOR_COLOR = "#9b59b6"  # Rich purple
    WORKER_COLOR = "#e67e22"      # Vibrant orange
    
    # Font settings
    FONT_FAMILY = "Consolas"
    FONT_SIZE = 10
    FONT_SIZE_SMALL = 9
    FONT_SIZE_LARGE = 12

class MinionTerminal:
    def __init__(self, root):
        """Initialize the minion terminal with modern UI elements"""
        self.root = root
        self.root.configure(bg=TerminalTheme.BG_COLOR)
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        self.root.title("🤖 Minions AI Terminal")
        
        # Set application icon
        try:
            self.root.iconbitmap("assets/minion_icon.ico")
        except:
            pass  # Icon file not found, continue without it
        
        # Create a loading animation canvas
        self.loading_canvas = tk.Canvas(
            self.root, 
            width=50, 
            height=50, 
            bg=TerminalTheme.BG_COLOR, 
            highlightthickness=0
        )
        self.loading_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.loading_dots = []
        
        # Create loading animation
        for i in range(8):
            angle = i * (360 / 8)
            x = 25 + 15 * (angle / 360 * 3.14 * 2)
            y = 25 + 15 * (angle / 360 * 3.14 * 2)
            dot = self.loading_canvas.create_oval(x-3, y-3, x+3, y+3, fill=TerminalTheme.ACCENT_COLOR)
            self.loading_dots.append(dot)
        
        # Hide loading animation initially
        self.loading_canvas.place_forget()
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Animation variables for thinking indicators
        self.thinking_animation_active = False
        self.thinking_dots_count = 0
        
        # Initialize UI components
        self.setup_ui()
        
        # Start output processing thread
        self.is_processing = True
        self.output_thread = threading.Thread(target=self.process_output)
        self.output_thread.daemon = True
        self.output_thread.start()
        
        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def on_close(self):
        """Handle window close event"""
        self.is_processing = False
        self.root.destroy()
    
    def setup_ui(self):
        """Set up the UI with modern styling"""
        # Create main frame with padding
        self.main_frame = tk.Frame(self.root, bg=TerminalTheme.BG_COLOR)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add title with icon
        self.title_frame = tk.Frame(self.main_frame, bg=TerminalTheme.BG_COLOR)
        self.title_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = tk.Label(
            self.title_frame,
            text="🤖 Minions AI Terminal",
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE_LARGE, "bold"),
            fg=TerminalTheme.TEXT_COLOR,
            bg=TerminalTheme.BG_COLOR
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Add status indicator
        self.status_frame = tk.Frame(self.title_frame, bg=TerminalTheme.BG_COLOR)
        self.status_frame.pack(side=tk.RIGHT)
        
        self.status_indicator = tk.Canvas(
            self.status_frame,
            width=12,
            height=12,
            bg=TerminalTheme.BG_COLOR,
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        
        # Draw status indicator (green dot)
        self.status_dot = self.status_indicator.create_oval(2, 2, 10, 10, fill=TerminalTheme.SUCCESS_COLOR)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Ready",
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE_SMALL),
            fg=TerminalTheme.SUCCESS_COLOR,
            bg=TerminalTheme.BG_COLOR
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Create toolbar with action buttons
        self.toolbar = tk.Frame(self.main_frame, bg=TerminalTheme.PANEL_COLOR, height=30)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Create toolbar buttons with improved icons
        self.clear_btn = self.create_toolbar_button(self.toolbar, "🗑️ Clear", self.clear_output)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.run_btn = self.create_toolbar_button(self.toolbar, "▶️ Run", self.start_minion_conversation)
        self.run_btn.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.save_btn = self.create_toolbar_button(self.toolbar, "💾 Save", self.save_conversation)
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.help_btn = self.create_toolbar_button(self.toolbar, "❓ Help", self.show_help)
        self.help_btn.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create output text area with syntax highlighting and line numbers
        self.output_frame = tk.Frame(self.main_frame, bg=TerminalTheme.PANEL_COLOR)
        self.output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers
        self.line_numbers = tk.Text(
            self.output_frame,
            width=4,
            padx=5,
            pady=5,
            bg=TerminalTheme.PANEL_COLOR,
            fg=TerminalTheme.TEXT_COLOR,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE),
            takefocus=0,
            bd=0,
            highlightthickness=0,
            state=tk.DISABLED
        )
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Main output text widget with scrollbar
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            bg=TerminalTheme.BG_COLOR,
            fg=TerminalTheme.TEXT_COLOR,
            insertbackground=TerminalTheme.TEXT_COLOR,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE),
            bd=0,
            highlightthickness=0
        )
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure tags for syntax highlighting
        self.output_text.tag_configure("prompt", foreground="#FF79C6")
        self.output_text.tag_configure("command", foreground="#50FA7B")
        self.output_text.tag_configure("output", foreground="#8BE9FD")
        self.output_text.tag_configure("error", foreground="#FF5555")
        self.output_text.tag_configure("highlight", background="#44475A", foreground="#F8F8F2")
        
        # Enhanced styling for thinking messages
        self.output_text.tag_configure(
            "supervisor_thinking", 
            foreground=TerminalTheme.SUPERVISOR_COLOR,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE+1, "bold"),
            background="#2a2a42"  # Subtle background highlight
        )
        
        self.output_text.tag_configure(
            "worker_thinking", 
            foreground=TerminalTheme.WORKER_COLOR,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE+1, "bold"),
            background="#2a3a2a"  # Subtle background highlight
        )
        
        self.output_text.tag_configure("user_message", foreground="#2ecc71")
        self.output_text.tag_configure("ai_message", foreground="#3498db")
        
        # Initialize ANSI colorizer
        self.colorizer = AnsiColorizer(self.output_text)
        
        # Create input area with improved styling
        self.input_frame = tk.Frame(self.main_frame, bg=TerminalTheme.BG_COLOR)
        self.input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.prompt_label = tk.Label(
            self.input_frame,
            text="💬 ",  # Chat bubble icon
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE, "bold"),
            fg=TerminalTheme.ACCENT_COLOR,
            bg=TerminalTheme.BG_COLOR,
            padx=5
        )
        self.prompt_label.pack(side=tk.LEFT)
        
        self.text_entry = tk.Text(
            self.input_frame,
            height=2,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE),
            bg=TerminalTheme.INPUT_BG,
            fg=TerminalTheme.TEXT_COLOR,
            insertbackground=TerminalTheme.TEXT_COLOR,
            bd=0,
            padx=10,
            pady=5,
            wrap=tk.WORD
        )
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.text_entry.insert("1.0", "Type a message...")
        self.text_entry.configure(fg=TerminalTheme.TEXT_COLOR)
        
        # Add placeholder behavior to text entry
        def on_text_focus_in(event):
            if self.text_entry.get("1.0", "end-1c").strip() == "Type a message...":
                self.text_entry.delete("1.0", tk.END)
                self.text_entry.configure(fg=TerminalTheme.TEXT_COLOR)
                
        def on_text_focus_out(event):
            if not self.text_entry.get("1.0", "end-1c").strip():
                self.text_entry.insert("1.0", "Type a message...")
                self.text_entry.configure(fg="#555555")
                
        self.text_entry.bind("<FocusIn>", on_text_focus_in)
        self.text_entry.bind("<FocusOut>", on_text_focus_out)
        self.text_entry.bind("<Return>", lambda e: self.send_message() if not (e.state & 0x0001) else None)
        
        # Create send button with improved styling
        self.send_btn = tk.Button(
            self.input_frame,
            text="📤 Send",  # Added send icon
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE),
            bg=TerminalTheme.BUTTON_BG,
            fg=TerminalTheme.TEXT_COLOR,
            activebackground=TerminalTheme.BUTTON_HOVER,
            activeforeground=TerminalTheme.TEXT_COLOR,
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=self.send_message
        )
        self.send_btn.pack(side=tk.LEFT, pady=2)
        
        # Add hover effect to send button
        def on_enter(e):
            self.send_btn['background'] = TerminalTheme.BUTTON_HOVER
            
        def on_leave(e):
            self.send_btn['background'] = TerminalTheme.BUTTON_BG
            
        self.send_btn.bind("<Enter>", on_enter)
        self.send_btn.bind("<Leave>", on_leave)
        
        # Status bar at bottom
        self.status_bar = tk.Label(
            self.main_frame,
            text="Ready to process commands",
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE_SMALL),
            fg=TerminalTheme.TEXT_COLOR,
            bg=TerminalTheme.PANEL_COLOR,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=5,
            pady=2
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
    def create_toolbar_button(self, parent, text, command):
        """Create a styled toolbar button"""
        button = tk.Button(
            parent,
            text=text,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE_SMALL),
            bg=TerminalTheme.BUTTON_BG,
            fg=TerminalTheme.TEXT_COLOR,
            activebackground=TerminalTheme.BUTTON_HOVER,
            activeforeground=TerminalTheme.TEXT_COLOR,
            relief=tk.FLAT,
            padx=5,
            pady=1,
            command=command
        )
        
        # Add hover effect
        def on_enter(e):
            button['background'] = TerminalTheme.BUTTON_HOVER
            
        def on_leave(e):
            button['background'] = TerminalTheme.BUTTON_BG
            
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button
    
    def start_minion_conversation(self):
        """Start a minion conversation with the current input"""
        # Get the text from the input field
        message = self.text_entry.get("1.0", "end-1c").strip()
        if message == "Type a message...":
            message = ""
            
        if not message:
            self.output_text.insert(tk.END, "Error: Please enter a message first\n", "error")
            self.output_text.see(tk.END)
            return
            
        # Clear the input field
        self.text_entry.delete("1.0", tk.END)
        self.text_entry.insert("1.0", "Type a message...")
        self.text_entry.configure(fg="#555555")
        
        # Update status and UI
        self.update_status("Processing", TerminalTheme.WARNING_COLOR)
        self.show_loading_animation()
        
        # Parse the message as if it were command line arguments
        class Args:
            def __init__(self, message):
                self.task = message
                self.full_messages = True
                self.no_color = True
                # Keep these additional attributes if needed
                self.model = None
                self.api_key = None
                self.system_prompt = None
                self.load_path = None
                self.save_path = None
                self.no_stream = False
                self.verbose = True
                
            def __str__(self):
                return self.task
            
            def __repr__(self):
                return f"Args(task='{self.task}')"
                
        args = Args(message)
        
        # Run minion in a separate thread
        threading.Thread(target=self.run_minion, args=(main, args), daemon=True).start()
        
    def update_status(self, text, color=None):
        """Update the status bar text and color"""
        self.status_bar.config(text=text)
        if color:
            self.status_label.config(text=text, fg=color)
            self.status_indicator.itemconfig(self.status_dot, fill=color)
    
    def show_loading_animation(self, show=True):
        """Show or hide the loading animation"""
        if show:
            self.loading_canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.update_loading_animation()
        else:
            self.loading_canvas.place_forget()
    
    def update_loading_animation(self):
        """Animate the loading indicator"""
        if not self.loading_canvas.winfo_exists():
            return
            
        # Rotate the colors of the dots
        colors = [self.loading_canvas.itemcget(dot, "fill") for dot in self.loading_dots]
        colors = [colors[-1]] + colors[:-1]  # Rotate
        
        # Apply the new colors
        for dot, color in zip(self.loading_dots, colors):
            self.loading_canvas.itemconfig(dot, fill=color)
            
        # Schedule the next update
        self.loading_canvas.after(100, self.update_loading_animation)
    
    def run_minion(self, task, args):
        """Run the minion task in a thread"""
        try:
            # Redirect stdout to catch output
            old_stdout = sys.stdout
            redirector = StdoutRedirector(output_queue)
            sys.stdout = redirector
            
            # Clear the output text
            self.output_text.delete("1.0", tk.END)
            
            # Add the command to the output
            self.output_text.insert(tk.END, f"$ Executing: {args.task}\n\n", "command")
            
            # Run the task with the provided arguments
            task(args)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Update status
            self.root.after(0, lambda: self.update_status("Command completed", TerminalTheme.SUCCESS_COLOR))
            self.root.after(0, lambda: self.show_loading_animation(False))
            
        except Exception as e:
            # Handle exceptions
            error_msg = f"Error executing command: {str(e)}\n"
            output_queue.put(error_msg)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Update status
            self.root.after(0, lambda: self.update_status("Error", TerminalTheme.ERROR_COLOR))
            self.root.after(0, lambda: self.show_loading_animation(False))
            
    def process_output(self):
        """Process output from the queue and update the text widget"""
        while self.is_processing:
            try:
                # Check if there's output to process
                try:
                    output = output_queue.get(block=False)
                    
                    # Check for special thinking messages with ANSI color codes
                    supervisor_pattern = r'Supervisor \(Remote\) is thinking\.\.\.'
                    worker_pattern = r'★ Worker \(Local\) is thinking\.\.\. ★'
                    
                    # Clean ANSI codes for detection and display
                    clean_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
                    
                    if re.search(supervisor_pattern, clean_output):
                        # Start thinking animation if not already running
                        if not self.thinking_animation_active:
                            self.thinking_animation_active = True
                            self.thinking_dots_count = 0
                            self.root.after(0, lambda: self.update_thinking_animation("supervisor"))
                        
                        # Display supervisor thinking message with enhanced styling
                        self.root.after(0, lambda: self.display_thinking_message(
                            "Supervisor (Remote) is thinking...", 
                            "supervisor_thinking"
                        ))
                    elif re.search(worker_pattern, clean_output):
                        # Start thinking animation if not already running
                        if not self.thinking_animation_active:
                            self.thinking_animation_active = True
                            self.thinking_dots_count = 0
                            self.root.after(0, lambda: self.update_thinking_animation("worker"))
                        
                        # Display worker thinking message with enhanced styling
                        self.root.after(0, lambda: self.display_thinking_message(
                            "★ Worker (Local) is thinking... ★", 
                            "worker_thinking"
                        ))
                    else:
                        # Normal output, use regular colorizer
                        self.root.after(0, lambda text=output: self.colorizer.apply_ansi_colors(text))
                        
                        # If we get normal output, stop the thinking animation
                        self.thinking_animation_active = False
                    
                    # Auto-scroll to see the latest output
                    self.root.after(0, lambda: self.output_text.see(tk.END))
                    
                except queue.Empty:
                    pass  # Queue is empty, continue
                    
                time.sleep(0.01)  # Small delay to reduce CPU usage
                
            except Exception as e:
                print(f"Error processing output: {e}")
                
    def send_message(self):
        """Send the message to be processed"""
        # Get the text from the input field
        message = self.text_entry.get("1.0", "end-1c").strip()
        if message == "Type a message..." or not message:
            return
            
        # Add the user message to the output with styling
        self.output_text.insert(tk.END, f"You: {message}\n", "user_message")
        self.output_text.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Start the minion conversation with the current input
        self.start_minion_conversation()
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.update_status("Output cleared", TerminalTheme.TEXT_COLOR)
        
        # Clear line numbers
        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete("1.0", tk.END)
        self.line_numbers.config(state=tk.DISABLED)

    def save_conversation(self):
        """Save the current conversation to a file"""
        try:
            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Conversation"
            )
            
            if file_path:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(self.output_text.get("1.0", tk.END))
                
                self.update_status(f"Conversation saved to {file_path}", TerminalTheme.SUCCESS_COLOR)
        except Exception as e:
            self.update_status(f"Error saving: {str(e)}", TerminalTheme.ERROR_COLOR)
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 Minions AI Terminal Help

- Type your message in the input box and press Send or Enter
- The AI will process your message and respond
- Special thinking messages will be highlighted in special colors
- Use toolbar buttons to clear, run, or save conversations

Example commands:
- Ask questions directly
- Request the AI to perform tasks
- Have conversations with the AI assistant

Tips:
- Press Ctrl+Enter to add a new line without sending
- Clear the output window to start a fresh conversation
        """
        
        # Create a popup window
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Minions AI Terminal")
        help_window.geometry("500x400")
        help_window.configure(bg=TerminalTheme.BG_COLOR)
        
        # Add help text
        help_text_widget = scrolledtext.ScrolledText(
            help_window,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            bg=TerminalTheme.BG_COLOR,
            fg=TerminalTheme.TEXT_COLOR,
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE),
            bd=0,
            highlightthickness=0
        )
        help_text_widget.pack(fill=tk.BOTH, expand=True)
        help_text_widget.insert(tk.END, help_text)
        help_text_widget.configure(state=tk.DISABLED)

    def display_thinking_message(self, message, tag):
        """Display thinking message with enhanced styling"""
        # Find and delete previous thinking message (if any)
        start_index = "1.0"
        while True:
            pos = self.output_text.search(message.split("...")[0], start_index, tk.END)
            if not pos:
                break
            
            line = self.output_text.get(pos, pos + " lineend")
            if message.split("...")[0] in line:
                self.output_text.delete(pos, pos + " lineend+1c")
            
            start_index = pos + "+1c"
        
        # Insert the thinking message with the appropriate tag
        self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)
    
    def update_thinking_animation(self, mode):
        """Update the animated dots for thinking messages"""
        if not self.thinking_animation_active:
            return
            
        # Define tag based on mode
        tag = "supervisor_thinking" if mode == "supervisor" else "worker_thinking"
        
        # Base message text
        if mode == "supervisor":
            base_msg = "Supervisor (Remote) is thinking"
        else:
            base_msg = "★ Worker (Local) is thinking"
        
        # Update dots animation
        self.thinking_dots_count = (self.thinking_dots_count + 1) % 4
        dots = "." * self.thinking_dots_count + " " * (3 - self.thinking_dots_count)
        
        # Complete message
        if mode == "supervisor":
            message = f"{base_msg}{dots}"
        else:
            message = f"{base_msg}{dots} ★"
        
        # Find and update existing message
        start_index = "1.0"
        updated = False
        
        while True and not updated:
            pos = self.output_text.search(base_msg, start_index, tk.END)
            if not pos:
                break
                
            line_end = self.output_text.index(f"{pos} lineend")
            self.output_text.delete(pos, line_end)
            self.output_text.insert(pos, message, tag)
            updated = True
            
            start_index = pos + "+1c"
        
        # If message wasn't found, add it
        if not updated:
            self.output_text.insert(tk.END, message + "\n", tag)
            self.output_text.see(tk.END)
        
        # Continue animation if still active
        if self.thinking_animation_active:
            self.root.after(300, lambda: self.update_thinking_animation(mode))
            
    # Add a method to stop thinking animation
    def stop_thinking_animation(self):
        """Stop the thinking animation"""
        self.thinking_animation_active = False

if __name__ == "__main__":
    # Create and run the application
    root = tk.Tk()
    app = MinionTerminal(root)
    root.mainloop() 