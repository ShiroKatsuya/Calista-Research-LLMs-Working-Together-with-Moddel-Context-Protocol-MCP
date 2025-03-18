import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import time
import io
import threading
import queue
import re
from contextlib import redirect_stdout, redirect_stderr
from main import Colors, colorize, main

# Theme constants for consistent styling
class TerminalTheme:
    BG_COLOR = "#1E1E1E"
    PANEL_COLOR = "#252526"
    TEXT_COLOR = "#E9E9E9"
    ACCENT_COLOR = "#007ACC"
    HIGHLIGHT_COLOR = "#264F78"
    ERROR_COLOR = "#F44747"
    SUCCESS_COLOR = "#FFFFFF"
    WARNING_COLOR = "#CCA700"
    INPUT_BG = "#3C3C3C"
    BUTTON_BG = "#0E639C"
    BUTTON_HOVER = "#1177BB"
    
    # Font settings with improved emoji support
    FONT_FAMILY = "Segoe UI Emoji"  # Primary font with good emoji support
    FONT_SIZE = 10
    FONT_SIZE_SMALL = 9
    FONT_SIZE_LARGE = 12
    
    # For monospace text (where needed)
    MONOSPACE_FONT = "Consolas"

# Custom stream class to enable real-time output
class QueuedStream(io.StringIO):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def write(self, string):
        super().write(string)
        # Push content to the queue immediately for real-time streaming
        self.queue.put(string)
        
    def flush(self):
        super().flush()

class MinionTerminal:
    def __init__(self, root):
        """Initialize the minion terminal with modern UI elements"""
        self.root = root
        self.root.configure(bg=TerminalTheme.BG_COLOR)
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        self.root.title("🤖 Minions AI Terminal")
  
        
        # Check and setup available fonts
        self._setup_available_fonts()
        
        # Try to set application icon
        try:
            self.root.iconbitmap("assets/minion_icon.ico")
        except:
            pass  # Icon file not found, continue without it
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Flag to track if processing is occurring
        self.is_processing = False
        
        # Initialize UI components
        self.setup_ui()
        
        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start the periodic check for new output
        self.check_output_queue()
        
    def _setup_available_fonts(self):
        """Check for available fonts and set up appropriate fallbacks"""
        available_fonts = list(tk.font.families())
        
        # Create a list of potential emoji fonts in order of preference
        emoji_fonts = ["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", 
                       "Segoe UI Symbol", "Symbola", "Arial Unicode MS"]
        
        # Find the first available emoji font
        emoji_font = next((font for font in emoji_fonts if font in available_fonts), "TkDefaultFont")
        
        # Set appropriate font in TerminalTheme
        TerminalTheme.FONT_FAMILY = emoji_font
        
        # For monospace, check availability
        monospace_fonts = ["Consolas", "Courier New", "DejaVu Sans Mono", "Menlo", "Monaco"]
        TerminalTheme.MONOSPACE_FONT = next((font for font in monospace_fonts if font in available_fonts), "TkFixedFont")
    
    def on_close(self):
        """Handle window close event"""
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
            fg=TerminalTheme.SUCCESS_COLOR,
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
            font=(TerminalTheme.MONOSPACE_FONT, TerminalTheme.FONT_SIZE_SMALL),
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
            font=(TerminalTheme.MONOSPACE_FONT, TerminalTheme.FONT_SIZE),
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
            font=(TerminalTheme.MONOSPACE_FONT, TerminalTheme.FONT_SIZE),
            bd=0,
            highlightthickness=0
        )
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Display welcome message
        self.output_text.insert(tk.END, "🤖 ", "emoji")
        self.output_text.insert(tk.END, "Welcome to Minions Terminal", "red")
        
        # Setup text tags for ANSI colors
        self.setup_ansi_tags()
        
        # Create input area with improved styling
        self.input_frame = tk.Frame(self.main_frame, bg=TerminalTheme.BG_COLOR)
        self.input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.prompt_label = tk.Label(
            self.input_frame,
            text="💬 ",  # Chat bubble icon
            font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE, "bold"),
            fg=TerminalTheme.SUCCESS_COLOR,
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
            fg=TerminalTheme.SUCCESS_COLOR,
            activebackground=TerminalTheme.BUTTON_HOVER,
            activeforeground=TerminalTheme.SUCCESS_COLOR,
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
            fg=TerminalTheme.SUCCESS_COLOR,
            activebackground=TerminalTheme.BUTTON_HOVER,
            activeforeground=TerminalTheme.SUCCESS_COLOR,
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
    
    def update_status(self, text, color=None):
        """Update the status bar text and color"""
        self.status_bar.config(text=text)
        if color:
            self.status_label.config(text=text, fg=color)
            self.status_indicator.itemconfig(self.status_dot, fill=color)
    
    def setup_ansi_tags(self):
        """Setup text tags to handle ANSI color codes"""
        # Basic ANSI colors
        self.output_text.tag_configure("black", foreground="#000000")
        self.output_text.tag_configure("red", foreground="#FF0000")
        self.output_text.tag_configure("green", foreground="#00FF00")
        self.output_text.tag_configure("yellow", foreground="#FFFF00")
        self.output_text.tag_configure("blue", foreground="#0000FF")
        self.output_text.tag_configure("magenta", foreground="#FF00FF")
        self.output_text.tag_configure("cyan", foreground="#00FFFF")
        self.output_text.tag_configure("white", foreground="#FFFFFF")
        
        # Bright colors
        self.output_text.tag_configure("bright_black", foreground="#555555")
        self.output_text.tag_configure("bright_red", foreground="#FF5555")
        self.output_text.tag_configure("bright_green", foreground="#55FF55")
        self.output_text.tag_configure("bright_yellow", foreground="#FFFF55")
        self.output_text.tag_configure("bright_blue", foreground="#5555FF")
        self.output_text.tag_configure("bright_magenta", foreground="#FF55FF")
        self.output_text.tag_configure("bright_cyan", foreground="#55FFFF")
        self.output_text.tag_configure("bright_white", foreground="#FFFFFF")
        
        # Background colors
        self.output_text.tag_configure("bg_black", background="#000000")
        self.output_text.tag_configure("bg_red", background="#FF0000")
        self.output_text.tag_configure("bg_green", background="#00FF00")
        self.output_text.tag_configure("bg_yellow", background="#FFFF00")
        self.output_text.tag_configure("bg_blue", background="#0000FF")
        self.output_text.tag_configure("bg_magenta", background="#FF00FF")
        self.output_text.tag_configure("bg_cyan", background="#00FFFF")
        self.output_text.tag_configure("bg_white", background="#FFFFFF")
        
        # Text styles
        self.output_text.tag_configure("bold", font=(TerminalTheme.MONOSPACE_FONT, TerminalTheme.FONT_SIZE, "bold"))
        self.output_text.tag_configure("italic", font=(TerminalTheme.MONOSPACE_FONT, TerminalTheme.FONT_SIZE, "italic"))
        self.output_text.tag_configure("underline", underline=1)
        
        # Special tag for emoji characters to ensure proper font rendering
        self.output_text.tag_configure("emoji", font=(TerminalTheme.FONT_FAMILY, TerminalTheme.FONT_SIZE))
    
    def parse_ansi_and_insert(self, text):
        """Parse ANSI escape codes and insert text with appropriate tags"""
        # Enhanced ANSI escape sequence pattern - capture more sequence types
        ansi_pattern = re.compile(r'\033\[([\d;]*)([a-zA-Z])')
        # Pattern to detect emoji characters
        emoji_pattern = re.compile(r'['
                                  r'\U0001F300-\U0001F64F'
                                  r'\U0001F680-\U0001F6FF'
                                  r'\u2600-\u26FF\u2700-\u27BF]+')
        
        # Current position in the text
        current_pos = 0
        
        # Current active tags
        active_tags = []
        
        # Process each ANSI code
        for match in ansi_pattern.finditer(text):
            # Insert text before the ANSI code
            pre_text = text[current_pos:match.start()]
            if pre_text:
                self._insert_with_emoji_detection(pre_text, active_tags)
            
            # Update current position
            current_pos = match.end()
            
            # Process the ANSI code
            code = match.group(1)
            command = match.group(2)
            
            # Only process 'm' commands (SGR - Select Graphic Rendition)
            if command == 'm':
                # Handle multiple codes separated by semicolons
                if not code:
                    # \033[m is equivalent to \033[0m (reset)
                    active_tags = []
                    continue
                    
                for subcode in code.split(';'):
                    if not subcode:
                        continue
                        
                    subcode = int(subcode)
                    
                    if subcode == 0:
                        # Reset all formatting
                        active_tags = []
                    elif subcode == 1:
                        active_tags.append("bold")
                    elif subcode == 3:
                        active_tags.append("italic")
                    elif subcode == 4:
                        active_tags.append("underline")
                    elif subcode == 30:
                        active_tags.append("black")
                    elif subcode == 31:
                        active_tags.append("red")
                    elif subcode == 32:
                        active_tags.append("green")
                    elif subcode == 33:
                        active_tags.append("yellow")
                    elif subcode == 34:
                        active_tags.append("blue")
                    elif subcode == 35:
                        active_tags.append("magenta")
                    elif subcode == 36:
                        active_tags.append("cyan")
                    elif subcode == 37:
                        active_tags.append("white")
                    # Background colors
                    elif subcode == 40:
                        active_tags.append("bg_black")
                    elif subcode == 41:
                        active_tags.append("bg_red")
                    elif subcode == 42:
                        active_tags.append("bg_green")
                    elif subcode == 43:
                        active_tags.append("bg_yellow")
                    elif subcode == 44:
                        active_tags.append("bg_blue")
                    elif subcode == 45:
                        active_tags.append("bg_magenta")
                    elif subcode == 46:
                        active_tags.append("bg_cyan")
                    elif subcode == 47:
                        active_tags.append("bg_white")
                    # Bright colors (90-97)
                    elif 90 <= subcode <= 97:
                        bright_colors = ["bright_black", "bright_red", "bright_green", 
                                         "bright_yellow", "bright_blue", "bright_magenta", 
                                         "bright_cyan", "bright_white"]
                        active_tags.append(bright_colors[subcode - 90])
        
        # Insert any remaining text
        remaining_text = text[current_pos:]
        if remaining_text:
            self._insert_with_emoji_detection(remaining_text, active_tags)
            
    def _insert_with_emoji_detection(self, text, active_tags):
        """Helper method to insert text with special handling for emoji characters"""
        # Pattern to detect emoji characters
        emoji_pattern = re.compile(r'['
                                  r'\U0001F300-\U0001F64F'
                                  r'\U0001F680-\U0001F6FF'
                                  r'\u2600-\u26FF\u2700-\u27BF]+')
        
        # Current position for text processing
        pos = 0
        
        # Find all emoji matches
        for match in emoji_pattern.finditer(text):
            # Insert text before emoji with regular tags
            if match.start() > pos:
                self.output_text.insert(tk.END, text[pos:match.start()], tuple(active_tags))
            
            # Insert emoji with emoji tag plus any active color/style tags
            emoji_tags = list(active_tags) + ["emoji"]
            self.output_text.insert(tk.END, match.group(0), tuple(emoji_tags))
            
            # Update position
            pos = match.end()
        
        # Insert any remaining text
        if pos < len(text):
            self.output_text.insert(tk.END, text[pos:], tuple(active_tags))
    
    def check_output_queue(self):
        """Check for new output from the processing thread"""
        try:
            while True:
                output = self.output_queue.get_nowait()
                if output:
                    # Use ANSI parsing instead of direct insertion
                    self.parse_ansi_and_insert(output)
                    self.output_text.see(tk.END)
                    
                    # Only add distinct system messages to conversation history
                    if output.strip():
                        self.conversation_history.append({"role": "system", "content": output})
                    
                    # Update line numbers if needed
                    self.update_line_numbers()
        except queue.Empty:
            pass
        
        # Schedule to check again after 50ms (more frequent for better streaming)
        self.root.after(50, self.check_output_queue)
    
    def update_line_numbers(self):
        """Update line numbers in the output text area"""
        if not hasattr(self, 'line_numbers'):
            return
            
        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete("1.0", tk.END)
        
        # Get the number of lines in the output text
        num_lines = int(self.output_text.index('end-1c').split('.')[0])
        
        # Add line numbers
        for i in range(1, num_lines + 1):
            self.line_numbers.insert(tk.END, f"{i}\n")
            
        self.line_numbers.config(state=tk.DISABLED)
    
    def process_command(self, message):
        """Process the command in a separate thread"""
        # Create a stream that will queue output in real-time
        stdout_stream = QueuedStream(self.output_queue)
        stderr_stream = QueuedStream(self.output_queue)
        
        # Capture the output from main function with real-time streaming
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            try:
                # Call the main function with the message
                result = main(message)
                
                # If main returns a value but doesn't print it, print it
                if result is not None and not stdout_stream.getvalue().strip():
                    print(result)
                    
                # Ensure all output is flushed
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception as e:
                print(f"Error: {str(e)}")
                sys.stdout.flush()
                sys.stderr.flush()
        
        # Signal that processing is complete
        self.root.after(0, self.processing_complete)
    
    def processing_complete(self):
        """Called when processing is complete"""
        self.is_processing = False
        self.update_status("Command executed", TerminalTheme.SUCCESS_COLOR)
        self.send_btn.config(state=tk.NORMAL)
        self.text_entry.config(state=tk.NORMAL)
    
    def send_message(self):
        """Send the message to be processed and capture output"""
        # If already processing, don't allow another command
        if self.is_processing:
            return
            
        # Get the text from the input field
        message = self.text_entry.get("1.0", "end-1c").strip()
        if message == "Type a message..." or not message:
            return
        
        # Set processing flag
        self.is_processing = True
        
        # Update UI to show processing state
        self.update_status("Processing...", TerminalTheme.ACCENT_COLOR)
        self.send_btn.config(state=tk.DISABLED)
        self.text_entry.config(state=tk.DISABLED)
            
        # Add user message to output with emoji support
        self._insert_with_emoji_detection(f"\n> {message}\n\n", [])
        self.output_text.see(tk.END)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_command, args=(message,))
        process_thread.daemon = True
        process_thread.start()
        
        # Clear the input field
        self.text_entry.delete("1.0", tk.END)
        self.text_entry.insert("1.0", "Type a message...")
        self.text_entry.configure(fg="#555555")
    
    def clear_output(self):
        """Clear the output text area"""
        self.output_text.delete("1.0", tk.END)
        self.update_status("Output cleared", TerminalTheme.TEXT_COLOR)
        
        # Clear line numbers
        self.line_numbers.config(state=tk.NORMAL)
        self.line_numbers.delete("1.0", tk.END)
        self.line_numbers.config(state=tk.DISABLED)

    def save_conversation(self):
        """Save the current conversation to a file"""
        try:
            import datetime
            from tkinter import filedialog
            
            # Generate default filename
            default_filename = f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            # Open file dialog
            filename = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            # If user cancels the dialog
            if not filename:
                return
                
            # Save the conversation
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.output_text.get("1.0", tk.END))
                
            self.update_status(f"Conversation saved to {filename}", TerminalTheme.SUCCESS_COLOR)
        except Exception as e:
            self.update_status(f"Error saving: {str(e)}", TerminalTheme.ERROR_COLOR)
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 Minions AI Terminal Help

- Type your message in the input box and press Send or Enter
- Messages and responses will be displayed in the output area
- The terminal is integrated into the GUI
- Output is streamed in real-time

Tips:
- Press Ctrl+Enter to add a new line without sending
- Use the Clear button to clear the output area
- Save button will save the conversation to a text file
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


if __name__ == "__main__":
    # Create and run the application
    root = tk.Tk()
    app = MinionTerminal(root)
    root.mainloop() 