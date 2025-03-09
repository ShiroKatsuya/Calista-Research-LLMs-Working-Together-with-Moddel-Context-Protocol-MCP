import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from message_handlers import MessageHandlers
import tkinter.messagebox as messagebox

class UIComponents:
    @staticmethod
    def setup_ui(app):
        """Initialize all UI components for the app."""
        # Main configuration
        app.root.configure(bg="#121B22")
        app.root.geometry("400x650")
        app.root.title(f"AI Voice Call - {app.model_label}")
        
        # Set app icon if available
        try:
            app.root.iconbitmap("assets/app_icon.ico")
        except:
            pass  # Ignore if icon not available
        
        # Main frame with subtle gradient effect
        app.main_frame = tk.Frame(app.root, bg="#121B22")
        app.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup UI sections
        UIComponents.setup_top_bar(app)
        UIComponents.setup_profile_section(app)
        UIComponents.setup_status_section(app)
        UIComponents.setup_prompt_section(app)
        UIComponents.setup_control_buttons(app)
    
    @staticmethod
    def setup_top_bar(app):
        """Set up the top bar of the app."""
        app.top_frame = tk.Frame(app.main_frame, bg="#1F2C34", height=60)
        app.top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a more modern top bar with lock icon
        lock_frame = tk.Frame(app.top_frame, bg="#1F2C34")
        lock_frame.pack(pady=(5, 0))
        
        # Create a canvas for the lock icon
        lock_canvas = tk.Canvas(lock_frame, width=20, height=20, bg="#1F2C34", highlightthickness=0)
        lock_canvas.pack(side=tk.LEFT, padx=(0, 5))
        lock_canvas.create_oval(3, 3, 17, 17, outline="#8696A0", width=1.5)
        lock_canvas.create_rectangle(7, 8, 13, 17, fill="#1F2C34", outline="#1F2C34")
        lock_canvas.create_line(10, 8, 10, 12, fill="#8696A0", width=1.5)
        
        app.secure_label = tk.Label(
            lock_frame, 
            text="End-to-end encrypted", 
            font=("Segoe UI", 10), 
            fg="#8696A0", 
            bg="#1F2C34"
        )
        app.secure_label.pack(side=tk.LEFT)
        
        # Add menu button to top-right
        menu_btn = tk.Canvas(app.top_frame, width=24, height=24, bg="#1F2C34", highlightthickness=0)
        menu_btn.pack(side=tk.RIGHT, padx=10, pady=5)
        # Draw three dots vertically
        for i in range(3):
            menu_btn.create_oval(10, 5+i*7, 14, 9+i*7, fill="#8696A0", outline="")
        menu_btn.bind("<Button-1>", lambda e: UIComponents._show_menu(app))

    @staticmethod
    def _show_menu(app):
        """Display a dropdown menu."""
        menu = tk.Menu(app.root, tearoff=0, bg="#262D31", fg="white", 
                      activebackground="#00BFA5", activeforeground="white",
                      font=("Segoe UI", 10))
        menu.add_command(label="Settings", command=lambda: None)
        menu.add_command(label="Clear Chat", command=app.clear_response)
        menu.add_separator()
        menu.add_command(label="About", command=lambda: None)
        menu.add_command(label="Help", command=lambda: None)
        
        # Get current mouse position relative to the root window
        x = app.root.winfo_pointerx() - app.root.winfo_rootx()
        y = app.root.winfo_pointery() - app.root.winfo_rooty()
        menu.post(app.root.winfo_rootx() + x, app.root.winfo_rooty() + y)

    @staticmethod
    def setup_profile_section(app):
        """Set up the profile section with avatar and model info."""
        app.profile_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.profile_frame.pack(pady=30)
        
        # Create a more professional profile avatar with gradient
        app.profile_circle = tk.Canvas(
            app.profile_frame, 
            width=150, 
            height=150, 
            bg="#121B22", 
            highlightthickness=0
        )
        app.profile_circle.pack()
        
        # Create outer glow effect
        app.profile_circle.create_oval(5, 5, 145, 145, fill="#128C7E", outline="")
        # Create gradient effect
        for i in range(5):
            app.profile_circle.create_oval(
                10+i, 10+i, 140-i, 140-i, 
                fill="", 
                outline="#25D366", 
                width=1.5-i*0.2
            )
        
        # Create interactive avatar with hover effect
        app.profile_circle.create_oval(20, 20, 130, 130, fill="#0E6655", outline="")
        app.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
        # Make avatar interactive with hover effect
        def on_avatar_enter(e):
            app.profile_circle.create_oval(20, 20, 130, 130, fill="#0C7D5A", outline="", tags="hover")
            app.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white", tags="hover")
        
        def on_avatar_leave(e):
            app.profile_circle.delete("hover")
        
        app.profile_circle.bind("<Enter>", on_avatar_enter)
        app.profile_circle.bind("<Leave>", on_avatar_leave)
        
        app.name_label = tk.Label(
            app.profile_frame, 
            text=app.model_label, 
            font=("Segoe UI", 24, "bold"), 
            fg="white", 
            bg="#121B22"
        )
        app.name_label.pack(pady=(15, 5))
        
        app.model_info_label = tk.Label(
            app.profile_frame, 
            text=app.model_name, 
            font=("Segoe UI", 14), 
            fg="#00BFA5", 
            bg="#121B22"
        )
        app.model_info_label.pack()

    @staticmethod
    def setup_status_section(app):
        """Set up the status section with labels and call buttons."""
        app.status_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.status_frame.pack(pady=10)
        
        # Status indicator with animated dots
        status_container = tk.Frame(app.status_frame, bg="#121B22")
        status_container.pack()
        
        # Add a small indicator dot
        app.status_indicator = tk.Canvas(status_container, width=12, height=12, 
                                        bg="#121B22", highlightthickness=0)
        app.status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        app.status_indicator.create_oval(2, 2, 10, 10, fill="#8696A0", tags="indicator")
        
        app.status_label = tk.Label(
            status_container, 
            font=("Segoe UI", 14), 
            fg="#8696A0", 
            bg="#121B22"
        )
        app.status_label.pack(side=tk.LEFT)
        
        # Duration with clock icon
        duration_container = tk.Frame(app.status_frame, bg="#121B22")
        duration_container.pack(pady=5)
        
        # Clock icon
        clock = tk.Canvas(duration_container, width=14, height=14, 
                         bg="#121B22", highlightthickness=0)
        clock.pack(side=tk.LEFT, padx=(0, 5))
        clock.create_oval(1, 1, 13, 13, outline="#8696A0", width=1)
        # Clock hands
        clock.create_line(7, 7, 7, 3, fill="#8696A0", width=1)
        clock.create_line(7, 7, 10, 7, fill="#8696A0", width=1)
        
        app.duration_label = tk.Label(
            duration_container, 
            text="", 
            font=("Segoe UI", 12), 
            fg="#8696A0", 
            bg="#121B22"
        )
        app.duration_label.pack(side=tk.LEFT)
        
        # Modern styled buttons
        app.start_call_button = tk.Button(
            app.status_frame, 
            text="Start Calling", 
            font=("Segoe UI", 12), 
            bg="#00BFA5", 
            fg="white",
            command=lambda: UIComponents.send_to_minion_terminal(app), 
            padx=15, 
            pady=6, 
            relief=tk.FLAT, 
            bd=0,
            activebackground="#00AD96",
            activeforeground="white"
        )
        app.start_call_button.pack(pady=10)
        
        # Add hover effect to call button
        def on_call_enter(e):
            app.start_call_button.config(bg="#00AD96")
        
        def on_call_leave(e):
            app.start_call_button.config(bg="#00BFA5")
        
        app.start_call_button.bind("<Enter>", on_call_enter)
        app.start_call_button.bind("<Leave>", on_call_leave)
        
        app.stop_call_button = tk.Button(
            app.status_frame, 
            text="Stop Calling", 
            font=("Segoe UI", 12), 
            bg="#FF0000", 
            fg="white",
            command=app.end_call, 
            padx=15, 
            pady=6, 
            relief=tk.FLAT, 
            bd=0,
            activebackground="#CC0000",
            activeforeground="white"
        )

    @staticmethod
    def send_to_minion_terminal(app):
        """Get text from input field and send it directly to minion_terminal.py"""
        # Get message from text entry
        if hasattr(app, 'text_entry'):
            message = app.text_entry.get("1.0", "end-1c").strip()
            
            # Skip placeholder text
            if message == "Type a message...":
                message = ""
                
            # Check if there's actually text to process
            if not message:
                messagebox.showwarning("Input Required", "Please type a message before starting the call.")
                return
                
            # Send the message to minion_terminal
            if hasattr(app.__class__, 'minion_terminal'):
                # Set the task entry in minion_terminal to this message
                app.__class__.minion_terminal.task_entry.delete(0, tk.END)
                app.__class__.minion_terminal.task_entry.insert(0, message)
                
                # Start the minion conversation
                app.__class__.minion_terminal.start_minion_conversation()
                
                # Clear the input field after sending
                app.text_entry.delete("1.0", tk.END)
                app.text_entry.insert("1.0", "Type a message...")
                app.text_entry.config(fg="#8696A0")
                
                # Update status
                if hasattr(app, 'status_label'):
                    app.status_label.config(text=f"{MessageHandlers.SEND_EMOJI} Message sent to Minion")
            else:
                messagebox.showerror("Error", "Minion terminal not initialized.")

    @staticmethod
    def setup_prompt_section(app):
        """Set up the conversation and text entry areas."""
        app.prompt_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Response area with modern header and action buttons
        app.response_frame = tk.Frame(app.prompt_frame, bg="#121B22")
        app.response_frame.pack(fill=tk.BOTH, expand=True)
        
        # Conversation header with actions
        header_frame = tk.Frame(app.response_frame, bg="#121B22")
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        app.response_label = tk.Label(
            header_frame, 
            text="Conversation", 
            font=("Segoe UI", 12, "bold"), 
            fg="white", 
            bg="#121B22", 
            anchor="w"
        )
        app.response_label.pack(side=tk.LEFT, fill=tk.X)
        
        # Add action buttons to conversation header
        search_btn = tk.Label(
            header_frame,
            text="🔍",
            font=("Segoe UI", 12),
            fg="#8696A0",
            bg="#121B22"
        )
        search_btn.pack(side=tk.RIGHT, padx=3)
        
        clear_btn = tk.Label(
            header_frame,
            text="🗑️",
            font=("Segoe UI", 12),
            fg="#8696A0",
            bg="#121B22"
        )
        clear_btn.pack(side=tk.RIGHT, padx=3)
        clear_btn.bind("<Button-1>", lambda e: app.clear_response())
        
        # Add hover effect to buttons
        for btn in [search_btn, clear_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg="#FFFFFF"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg="#8696A0"))
        
        # Response text widget with custom scrollbar
        response_scroll = ttk.Scrollbar(app.response_frame)
        response_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a style for the scrollbar
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Custom.Vertical.TScrollbar", 
                       gripcount=0, 
                       background="#2A343B", 
                       troughcolor="#1F2C34",
                       borderwidth=0,
                       arrowsize=14)
        response_scroll.config(style="Custom.Vertical.TScrollbar")
        
        app.response_text = tk.Text(
            app.response_frame, 
            wrap=tk.WORD, 
            bg="#1F2C34", 
            fg="white", 
            insertbackground="white", 
            height=10, 
            font=("Segoe UI Emoji", 11),  # Use Emoji-compatible font
            relief=tk.FLAT, 
            padx=15,  # Increased padding
            pady=15,
            spacing1=4,  # Space above each line
            spacing2=3,  # Space between lines
            spacing3=4   # Space after each paragraph
        )
        app.response_text.pack(fill=tk.BOTH, expand=True)
        app.response_text.config(yscrollcommand=response_scroll.set)
        response_scroll.config(command=app.response_text.yview)
        
        # Configure text tags for different message types
        # Message bubbles with better visual styling
        app.response_text.tag_configure("worker_message", 
                                       foreground="#FFFFFF", 
                                       background="#2A5343",  # Green background for Worker
                                       lmargin1=25, 
                                       lmargin2=25, 
                                       rmargin=10,
                                       relief=tk.FLAT,
                                       borderwidth=0,
                                       font=("Segoe UI Emoji", 11),
                                       spacing1=4,
                                       spacing3=4)
                                       
        app.response_text.tag_configure("supervisor_message", 
                                       foreground="#FFFFFF", 
                                       background="#2A4153",  # Blue background for Supervisor
                                       lmargin1=25, 
                                       lmargin2=25, 
                                       rmargin=10,
                                       relief=tk.FLAT,
                                       borderwidth=0,
                                       font=("Segoe UI Emoji", 11),
                                       spacing1=4,
                                       spacing3=4)
        
        # System message styling
        app.response_text.tag_configure("system_message", 
                                       foreground="#FFFFFF", 
                                       background="#333333",  # Dark gray for system
                                       lmargin1=25, 
                                       lmargin2=25, 
                                       rmargin=10,
                                       relief=tk.FLAT,
                                       borderwidth=0,
                                       font=("Segoe UI Emoji", 10),
                                       spacing1=3,
                                       spacing3=3)
        
        # Call info for duration display
        app.response_text.tag_configure("call_info", 
                                       foreground="#8696A0",
                                       font=("Segoe UI", 10),
                                       spacing1=2,
                                       spacing3=5,
                                       lmargin1=20)
                                       
        # Interactive message action buttons
        app.response_text.tag_configure("message_action", 
                                       foreground="#8696A0",
                                       font=("Segoe UI Emoji", 10))
        app.response_text.tag_bind("message_action", "<Enter>", 
                                  lambda e: app.response_text.config(cursor="hand2"))
        app.response_text.tag_bind("message_action", "<Leave>", 
                                  lambda e: app.response_text.config(cursor=""))
                                       
        # Code block icon style
        app.response_text.tag_configure("code_icon", 
                                       foreground="#FFEB3B",
                                       font=("Segoe UI Emoji", 11))
                                       
        # Code block styling
        app.response_text.tag_configure("code_block", 
                                       foreground="#E6E1DC", 
                                       background="#232323",
                                       font=("Courier New", 10),
                                       lmargin1=30, 
                                       lmargin2=30, 
                                       rmargin=10,
                                       spacing1=5,
                                       spacing3=5)
                                       
        app.response_text.tag_configure("code_header", 
                                       foreground="#CCCCCC", 
                                       background="#333333",
                                       font=("Segoe UI", 9, "bold"),
                                       lmargin1=30, 
                                       spacing1=3)
                                       
        # Thinking indicator styling with animation effect
        app.response_text.tag_configure("worker_thinking", 
                                       foreground="#8696A0", 
                                       font=("Segoe UI Emoji", 11, "italic"),
                                       spacing1=2)
                                       
        app.response_text.tag_configure("supervisor_thinking", 
                                       foreground="#8696A0", 
                                       font=("Segoe UI Emoji", 11, "italic"),
                                       spacing1=2)
                                       
        # Enhanced header tags for better visibility
        app.response_text.tag_configure("worker_header", 
                                       foreground="#25D366", 
                                       font=("Segoe UI Emoji", 11, "bold"), 
                                       spacing1=8,
                                       lmargin1=10)
                                       
        app.response_text.tag_configure("supervisor_header", 
                                       foreground="#34B7F1", 
                                       font=("Segoe UI Emoji", 11, "bold"), 
                                       spacing1=8,
                                       lmargin1=10)
                                       
        app.response_text.tag_configure("system_header", 
                                       foreground="#FFEB3B", 
                                       font=("Segoe UI Emoji", 10, "bold"), 
                                       spacing1=5,
                                       lmargin1=10)
                                       
        # Legacy tags (keeping for backward compatibility)
        app.response_text.tag_configure("worker_question_header", foreground="#25D366", font=("Segoe UI Emoji", 11, "bold"))
        app.response_text.tag_configure("supervisor_question_header", foreground="#34B7F1", font=("Segoe UI Emoji", 11, "bold"))
        app.response_text.tag_configure("worker_answer_header", foreground="#25D366", font=("Segoe UI Emoji", 11, "bold"))
        app.response_text.tag_configure("supervisor_answer_header", foreground="#34B7F1", font=("Segoe UI Emoji", 11, "bold"))
        
        # Improved separator and message formatting
        app.response_text.tag_configure("waiting_message", 
                                       foreground="#8696A0", 
                                       font=("Segoe UI Emoji", 11, "italic"),
                                       spacing1=2,
                                       spacing3=2)
                                       
        app.response_text.tag_configure("separator", 
                                       foreground="#666666", 
                                       justify="center",
                                       spacing1=6, 
                                       spacing2=2, 
                                       spacing3=6)
                                       
        app.response_text.tag_configure("light_separator", 
                                       foreground="#444444", 
                                       justify="center",
                                       spacing1=3, 
                                       spacing2=1, 
                                       spacing3=5)
        
        # Auto-scroll to bottom when new content is added
        def fix_scrollbar(*args):
            # Only auto-scroll if already at or near the bottom
            try:
                view = app.response_text.yview()
                if view[1] > 0.9:  # If we're already near the bottom
                    app.response_text.yview_moveto(1.0)
            except:
                pass

        app.response_text.vbar = response_scroll
        app.response_text.vbar.configure(command=app.response_text.yview)
        app.response_text.configure(yscrollcommand=fix_scrollbar)
        
        # Text entry area with modern styling and attachments
        app.entry_frame = tk.Frame(app.prompt_frame, bg="#1F2C34", bd=0, relief=tk.FLAT)
        app.entry_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Attachments bar
        attachments_frame = tk.Frame(app.entry_frame, bg="#1F2C34", height=30)
        attachments_frame.pack(fill=tk.X)
        
        # Attachment buttons
        attachment_btn = tk.Label(
            attachments_frame,
            text="📎",
            font=("Segoe UI", 12),
            fg="#8696A0",
            bg="#1F2C34"
        )
        attachment_btn.pack(side=tk.LEFT, padx=10)
        
        camera_btn = tk.Label(
            attachments_frame,
            text="📷",
            font=("Segoe UI", 12),
            fg="#8696A0",
            bg="#1F2C34"
        )
        camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Add hover effect to buttons
        for btn in [attachment_btn, camera_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg="#FFFFFF"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(fg="#8696A0"))

        # Text input field
        app.text_entry = tk.Text(
            app.entry_frame, 
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
        app.text_entry.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        app.text_entry.insert("1.0", "Type a message...")
        app.text_entry.config(fg="#8696A0")
        
        # Clear placeholder on focus
        def on_text_focus_in(event):
            if app.text_entry.get("1.0", "end-1c") == "Type a message...":
                app.text_entry.delete("1.0", tk.END)
                app.text_entry.config(fg="white")
                
        def on_text_focus_out(event):
            if not app.text_entry.get("1.0", "end-1c").strip():
                app.text_entry.delete("1.0", tk.END)
                app.text_entry.insert("1.0", "Type a message...")
                app.text_entry.config(fg="#8696A0")
                
        app.text_entry.bind("<FocusIn>", on_text_focus_in)
        app.text_entry.bind("<FocusOut>", on_text_focus_out)
        
        # Send button frame
        send_frame = tk.Frame(app.entry_frame, bg="#1F2C34")
        send_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Send button
        send_button = tk.Button(
            send_frame,
            text=f"{MessageHandlers.SEND_EMOJI}",
            font=("Segoe UI Emoji", 14),
            bg="#00BFA5",
            fg="white",
            bd=0,
            padx=10,
            pady=5,
            relief=tk.FLAT,
            command=app.send_to_minion if hasattr(app.__class__, 'minion_terminal') and app.__class__.minion_terminal is not None else app.toggle_call,
            activebackground="#00AD96",
            activeforeground="white"
        )
        send_button.pack(side=tk.RIGHT)
        
        # Bind Enter key to send message
        def on_enter_key(event):
            if hasattr(app.__class__, 'minion_terminal') and app.__class__.minion_terminal is not None:
                # Get the message from the text entry
                message = app.text_entry.get("1.0", "end-1c").strip()
                if message and message != "Type a message...":
                    # Set the message in minion_terminal
                    app.__class__.minion_terminal.text_entry.delete("1.0", tk.END)
                    app.__class__.minion_terminal.text_entry.insert("1.0", message)
                    
                    # Send the message directly to minion_terminal
                    app.__class__.minion_terminal.send_message()
                    
                    # Clear the input field after sending
                    app.text_entry.delete("1.0", tk.END)
            else:
                app.toggle_call()
            return "break"  # Prevent default behavior
            
        app.text_entry.bind("<Return>", on_enter_key)
        
        # Help text
        help_text = tk.Label(
            send_frame,
            text="Press Enter to send",
            font=("Segoe UI", 9),
            fg="#8696A0",
            bg="#1F2C34"
        )
        help_text.pack(side=tk.LEFT, padx=5)

    @staticmethod
    def setup_control_buttons(app):
        """Set up the bottom control buttons."""
        app.controls_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        # Add history toggle button at top of controls
        app.history_frame = tk.Frame(app.controls_frame, bg="#121B22")
        app.history_frame.pack(pady=(0, 10))
        
        # Modern toggle switch appearance
        history_switch_frame = tk.Frame(app.history_frame, bg="#121B22", padx=5, pady=5)
        history_switch_frame.pack()
        
        # Label for toggle
        app.history_label = tk.Label(
            history_switch_frame,
            text="Save History:",
            font=("Segoe UI", 9),
            fg="#8696A0",
            bg="#121B22"
        )
        app.history_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # Custom toggle button
        app.history_button = tk.Button(
            history_switch_frame,
            text="OFF",
            font=("Segoe UI", 9, "bold"),
            bg="#262D31",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=3,
            activebackground="#1F2C34",
            activeforeground="#FFFFFF",
            command=lambda: UIComponents.toggle_history_btn(app)
        )
        app.history_button.pack(side=tk.LEFT)
        
        app.buttons_frame = tk.Frame(app.controls_frame, bg="#121B22")
        app.buttons_frame.pack(pady=20)
        
        button_size = 60
        app.mute_frame = UIComponents.create_circular_button(app, app.buttons_frame, button_size, "#262D31", "Mute")
        app.mute_frame.pack(side=tk.LEFT, padx=15)
        
        app.call_button_frame = UIComponents.create_circular_button(
            app,
            app.buttons_frame, 
            button_size + 20, 
            "#00BFA5", 
            "Call", 
            command=app.toggle_call
        )
        app.call_button_frame.pack(side=tk.LEFT, padx=15)
        
        app.speaker_frame = UIComponents.create_circular_button(app, app.buttons_frame, button_size, "#262D31", "Speaker")
        app.speaker_frame.pack(side=tk.LEFT, padx=15)

    @staticmethod
    def toggle_history_btn(app):
        """Toggle history preservation and update button appearance."""
        preserve = app.toggle_preserve_history()
        if preserve:
            app.history_button.config(
                text="ON",
                bg="#00BFA5"  # Green color when active
            )
        else:
            app.history_button.config(
                text="OFF",
                bg="#262D31"  # Default dark color when inactive
            )

    @staticmethod
    def create_circular_button(app, parent, size, color, text, command=None):
        """Create a circular button with the given parameters."""
        frame = tk.Frame(parent, bg="#121B22")
        
        button = tk.Canvas(frame, width=size, height=size, bg="#121B22", highlightthickness=0)
        button.pack()
        
        # Create outer glow effect
        for i in range(3):
            alpha = 0.3 - (i * 0.1)  # Decreasing alpha for outer glow
            glow_color = color
            if color == "#00BFA5":  # For call button - green glow
                button.create_oval(
                    0+i, 0+i, size-i, size-i, 
                    outline=glow_color, 
                    width=1,
                    tags="button_glow"
                )
        
        # Main button circle
        button.create_oval(2, 2, size - 2, size - 2, fill=color, outline="", width=0, tags="button_bg")
        
        # Button icon based on text
        if text == "Call":
            button.create_text(size // 2, size // 2, text="📞", font=("Segoe UI Emoji", size // 3), fill="white", tags="button_icon")
        elif text == "Mute":
            button.create_text(size // 2, size // 2, text="🎤", font=("Segoe UI Emoji", size // 3), fill="white", tags="button_icon")
        elif text == "Speaker":
            button.create_text(size // 2, size // 2, text="🔊", font=("Segoe UI Emoji", size // 3), fill="white", tags="button_icon")
        elif text == "End":
            button.create_text(size // 2, size // 2, text="📞", font=("Segoe UI Emoji", size // 3), fill="white", tags="button_icon")
        
        # Add hover and click effects
        def on_enter(e):
            nonlocal button, color
            # Lighter color on hover
            hover_color = "#00D1B2" if color == "#00BFA5" else "#333A40"
            button.itemconfig("button_bg", fill=hover_color)
            
        def on_leave(e):
            nonlocal button, color
            # Restore original color
            button.itemconfig("button_bg", fill=color)
            
        def on_click(e):
            nonlocal button, color
            # Darker color when clicked
            click_color = "#00A78E" if color == "#00BFA5" else "#1A1F23"
            button.itemconfig("button_bg", fill=click_color)
            if command:
                command()
            # Schedule restore of hover color after click
            button.after(200, lambda: button.itemconfig("button_bg", fill=color))
            
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        button.bind("<Button-1>", on_click)
        
        label = tk.Label(frame, text=text, font=("Segoe UI", 10), fg="#8696A0", bg="#121B22")
        label.pack(pady=(5, 0))
        
        return frame 