import tkinter as tk

class UIComponents:
    @staticmethod
    def setup_ui(app):
        """Initialize all UI components for the app."""
        # Main configuration
        app.root.configure(bg="#121B22")
        app.root.geometry("400x650")
        
        # Main frame
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
        app.secure_label = tk.Label(
            app.top_frame, 
            text="End-to-end encrypted", 
            font=("Segoe UI", 10), 
            fg="#8696A0", 
            bg="#1F2C34"
        )
        app.secure_label.pack(pady=(10, 0))

    @staticmethod
    def setup_profile_section(app):
        """Set up the profile section with avatar and model info."""
        app.profile_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.profile_frame.pack(pady=30)
        
        app.profile_circle = tk.Canvas(
            app.profile_frame, 
            width=150, 
            height=150, 
            bg="#121B22", 
            highlightthickness=0
        )
        app.profile_circle.pack()
        app.profile_circle.create_oval(10, 10, 140, 140, fill="#128C7E", outline="#25D366", width=2)
        app.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
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
        
        app.status_label = tk.Label(
            app.status_frame, 
            font=("Segoe UI", 14), 
            fg="#8696A0", 
            bg="#121B22"
        )
        app.status_label.pack()
        
        app.duration_label = tk.Label(
            app.status_frame, 
            text="", 
            font=("Segoe UI", 14), 
            fg="#8696A0", 
            bg="#121B22"
        )
        
        app.start_call_button = tk.Button(
            app.status_frame, 
            text="Start Calling", 
            font=("Segoe UI", 12), 
            bg="#00BFA5", 
            fg="white",
            command=app.toggle_call, 
            padx=10, 
            pady=5, 
            relief=tk.RAISED, 
            bd=0
        )
        app.start_call_button.pack(pady=10)
        
        app.stop_call_button = tk.Button(
            app.status_frame, 
            text="Stop Calling", 
            font=("Segoe UI", 12), 
            bg="#FF0000", 
            fg="white",
            command=app.end_call, 
            padx=10, 
            pady=5, 
            relief=tk.RAISED, 
            bd=0
        )

    @staticmethod
    def setup_prompt_section(app):
        """Set up the conversation and text entry areas."""
        app.prompt_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Response area (where conversation appears)
        app.response_frame = tk.Frame(app.prompt_frame, bg="#121B22")
        app.response_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label for the response area
        app.response_label = tk.Label(
            app.response_frame, 
            text="Conversation", 
            font=("Segoe UI", 12, "bold"), 
            fg="white", 
            bg="#121B22", 
            anchor="w"
        )
        app.response_label.pack(fill=tk.X, pady=(0, 5))
        
        # Response text widget with scrollbar
        response_scroll = tk.Scrollbar(app.response_frame)
        response_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
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
                                       
        app.response_text.tag_configure("timestamp", 
                                       foreground="#8696A0", 
                                       font=("Segoe UI Emoji", 9))
        
        # Fix for auto-scrolling - ensure scrollbar always shows at bottom with new content
        def fix_scrollbar(*args):
            # Only auto-scroll if already at or near the bottom
            first, last = args[0], args[1]
            if float(last) > 0.9:  # If we're already near the bottom
                app.response_text.see(tk.END)
                app.response_text.yview_moveto(1.0)
                
        # Override the default scrollbar behavior to enable smooth auto-scrolling
        app.response_text.config(yscrollcommand=lambda first, last: (
            response_scroll.set(first, last),
            fix_scrollbar(first, last)
        ))
        
        # Set initial readonly state
        app.response_text.config(state=tk.DISABLED)
        
        # Text entry area at the bottom
        app.entry_frame = tk.Frame(app.prompt_frame, bg="#1F2C34", height=100)
        app.entry_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add a label for the text entry area
        app.prompt_label = tk.Label(
            app.prompt_frame, 
            text="Your message:", 
            font=("Segoe UI", 12), 
            fg="#8696A0", 
            bg="#121B22", 
            anchor="w"
        )
        app.prompt_label.pack(fill=tk.X, pady=(10, 5))
        
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
        # Bind Return key to toggle_call
        app.text_entry.bind("<Return>", app.toggle_call)

    @staticmethod
    def setup_control_buttons(app):
        """Set up the bottom control buttons."""
        app.controls_frame = tk.Frame(app.main_frame, bg="#121B22")
        app.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        # Add history toggle button at top of controls
        app.history_frame = tk.Frame(app.controls_frame, bg="#121B22")
        app.history_frame.pack(pady=(0, 10))
        
        app.history_button = tk.Button(
            app.history_frame,
            text="Save History: OFF",
            font=("Segoe UI", 9),
            bg="#262D31",
            fg="#FFFFFF",
            bd=0,
            padx=10,
            pady=5,
            activebackground="#1F2C34",
            activeforeground="#FFFFFF",
            command=lambda: UIComponents.toggle_history_btn(app)
        )
        app.history_button.pack()
        
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
                text="Save History: ON",
                bg="#00BFA5"  # Green color when active
            )
        else:
            app.history_button.config(
                text="Save History: OFF",
                bg="#262D31"  # Default dark color when inactive
            )

    @staticmethod
    def create_circular_button(app, parent, size, color, text, command=None):
        """Create a circular button with the given parameters."""
        frame = tk.Frame(parent, bg="#121B22")
        
        button = tk.Canvas(frame, width=size, height=size, bg="#121B22", highlightthickness=0)
        button.pack()
        
        button.create_oval(2, 2, size - 2, size - 2, fill=color, outline="", width=0)
        
        if text == "Call":
            button.create_text(size // 2, size // 2, text="📞", font=("Segoe UI", size // 3), fill="white")
        elif text == "Mute":
            button.create_text(size // 2, size // 2, text="🎤", font=("Segoe UI", size // 3), fill="white")
        elif text == "Speaker":
            button.create_text(size // 2, size // 2, text="🔊", font=("Segoe UI", size // 3), fill="white")
        elif text == "End":
            button.create_text(size // 2, size // 2, text="📞", font=("Segoe UI", size // 3), fill="white")
        
        if command:
            button.bind("<Button-1>", lambda event: command())
        
        label = tk.Label(frame, text=text, font=("Segoe UI", 10), fg="#8696A0", bg="#121B22")
        label.pack(pady=(5, 0))
        
        return frame 