import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from message_handlers import MessageHandlers
import tkinter.messagebox as messagebox
import time
import threading

class UIComponents:
    # Animation constants
    ANIMATION_SPEED = 50  # milliseconds between animation frames
    ANIMATION_COLOR_ACTIVE = "#25D366"
    ANIMATION_COLOR_INACTIVE = "#8696A0"
    
    # UI Theme colors
    DARK_BG = "#121B22"
    DARKER_BG = "#0A1014"
    PANEL_BG = "#1F2C34"
    TEXT_COLOR = "#E9EDEF"
    HIGHLIGHT_COLOR = "#00A884"
    INACTIVE_COLOR = "#8696A0"
    MESSAGE_BG_SENT = "#005C4B"
    MESSAGE_BG_RECEIVED = "#1F2C34"
    
    @staticmethod
    def setup_ui(app):
        """Initialize all UI components for the app."""
        # Main configuration
        app.root.configure(bg=UIComponents.DARK_BG)
        app.root.geometry("400x650")
        app.root.title(f"AI Voice Call - {app.model_label}")
        
        # Set app icon if available
        try:
            app.root.iconbitmap("assets/app_icon.ico")
        except:
            pass  # Ignore if icon not available
        
        # Main frame with subtle gradient effect
        app.main_frame = tk.Frame(app.root, bg=UIComponents.DARK_BG)
        app.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup UI sections
        UIComponents.setup_top_bar(app)
        UIComponents.setup_profile_section(app)
        UIComponents.setup_status_section(app)
        UIComponents.setup_prompt_section(app)
        UIComponents.setup_control_buttons(app)
        
        # Add ripple effect to clickable elements
        UIComponents.add_ripple_effect(app)
    
    @staticmethod
    def add_ripple_effect(app):
        """Add ripple effect to clickable elements"""
        # This is a placeholder for actual ripple effect implementation
        # In a real implementation, this would add canvas overlays to buttons
        pass
    
    @staticmethod
    def animate_button_click(button, original_color):
        """Animate button click with color change"""
        highlight_color = UIComponents.HIGHLIGHT_COLOR
        
        def animate(count=0, direction=1):
            if count >= 5:  # Animation completed
                button.configure(bg=original_color)
                return
                
            # Calculate intermediate color based on progress
            r1, g1, b1 = button.winfo_rgb(original_color)
            r2, g2, b2 = button.winfo_rgb(highlight_color)
            
            # Linear interpolation between colors
            t = count / 5.0 if direction > 0 else (5 - count) / 5.0
            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t
            
            # Convert to hex color
            color = f"#{int(r/256):02x}{int(g/256):02x}{int(b/256):02x}"
            button.configure(bg=color)
            
            # Schedule next animation frame
            button.after(UIComponents.ANIMATION_SPEED, animate, count + 1, direction)
        
        # Start animation
        animate()
        
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
        """Set up the profile section of the app."""
        # Create a frame for profile with modern design
        app.profile_frame = tk.Frame(app.main_frame, bg=UIComponents.PANEL_BG)
        app.profile_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create circular avatar with border
        avatar_size = 60
        avatar_frame = tk.Frame(app.profile_frame, bg=UIComponents.PANEL_BG)
        avatar_frame.pack(pady=(10, 5))
        
        # Create a canvas for the circular avatar
        app.avatar_canvas = tk.Canvas(
            avatar_frame, 
            width=avatar_size, 
            height=avatar_size, 
            bg=UIComponents.PANEL_BG, 
            highlightthickness=0
        )
        app.avatar_canvas.pack()
        
        # Draw a circle for avatar placeholder
        app.avatar_canvas.create_oval(
            2, 2, avatar_size-2, avatar_size-2, 
            fill=UIComponents.MESSAGE_BG_SENT, 
            outline=UIComponents.HIGHLIGHT_COLOR, 
            width=2
        )
        
        # Add avatar icon or initials
        if app.model_label == "Worker (Local)":
            emoji = MessageHandlers.WORKER_EMOJI
            status = MessageHandlers.ONLINE_STATUS  # Online status
        else:
            emoji = MessageHandlers.SUPERVISOR_EMOJI
            status = MessageHandlers.ONLINE_STATUS
            
        app.avatar_canvas.create_text(
            avatar_size//2, 
            avatar_size//2, 
            text="RL", 
            font=("Segoe UI Emoji", 24)
        )
        
        # Add hover effect to avatar
        def on_avatar_enter(e):
            app.avatar_canvas.create_oval(
                2, 2, avatar_size-2, avatar_size-2, 
                fill=UIComponents.MESSAGE_BG_SENT, 
                outline=UIComponents.HIGHLIGHT_COLOR, 
                width=3,
                tags="hover"
            )
            
        def on_avatar_leave(e):
            app.avatar_canvas.delete("hover")
            
        app.avatar_canvas.bind("<Enter>", on_avatar_enter)
        app.avatar_canvas.bind("<Leave>", on_avatar_leave)
        
        # Animated status indicator
        status_frame = tk.Frame(app.profile_frame, bg=UIComponents.PANEL_BG)
        status_frame.pack(pady=(0, 5))
        
        # Add model name with status indicator
        name_label = tk.Label(
            status_frame, 
            text=f"SUPERVISOR (Remote)", 
            font=("Arial", 14, "bold"), 
            fg=UIComponents.TEXT_COLOR, 
            bg=UIComponents.PANEL_BG
        )
        name_label.pack()
        
        # Add status message
        app.status_label = tk.Label(
            app.profile_frame, 
            text="Ready for conversation", 
            font=("Segoe UI", 10), 
            fg=UIComponents.INACTIVE_COLOR, 
            bg=UIComponents.PANEL_BG
        )
        app.status_label.pack(pady=(0, 10))
        
        # Add subtle separator
        separator = ttk.Separator(app.main_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=10)

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
            
            # Add animation effect to the button
            original_bg = app.start_call_button.cget("bg")
            app.start_call_button.config(bg="#00796B")  # Darker green for animation effect
            
            # Add connecting animation to status indicator
            if hasattr(app, 'status_indicator'):
                app.status_indicator.itemconfig("indicator", fill="#FFC107")  # Yellow for connecting
                
                # Create pulsing effect for the button and status indicator
                def pulse_animation(count=0):
                    if count >= 6:  # 3 pulses (2 frames each)
                        # Restore original button color
                        app.start_call_button.config(bg=original_bg)
                        
                        # Send the message to minion_terminal
                        if hasattr(app.__class__, 'minion_terminal'):
                            # Set the text entry in minion_terminal to this message
                            app.__class__.minion_terminal.text_entry.delete("1.0", tk.END)
                            app.__class__.minion_terminal.text_entry.insert("1.0", message)
                            
                            # Start the minion conversation
                            app.__class__.minion_terminal.start_minion_conversation()
                            
                            # Clear the input field after sending
                            app.text_entry.delete("1.0", tk.END)
                            app.text_entry.insert("1.0", "Type a message...")
                            app.text_entry.config(fg="#8696A0")
                            
                            # Update status
                            if hasattr(app, 'status_label'):
                                app.status_label.config(text=f"{MessageHandlers.SEND_EMOJI} Message sent to Minion")
                            
                            # Also add a confirmation in the response area
                            if hasattr(app, 'response_text'):
                                app.response_text.config(state=tk.NORMAL)
                                # Add separator if needed
                                if MessageHandlers.preserve_history and app.response_text.get("1.0", tk.END).strip():
                                    app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
                                # Add confirmation message
                                app.response_text.insert(tk.END, f"{MessageHandlers.SEND_EMOJI} Message sent to Minion: \"{message}\"\n", "system_header")
                                app.response_text.config(state=tk.DISABLED)
                                MessageHandlers._ensure_autoscroll(app)
                        else:
                            messagebox.showerror("Error", "Minion terminal not initialized.")
                        return
                    
                    # Update animation colors (pulse effect)
                    current_fill = app.status_indicator.itemcget("indicator", "fill")
                    new_fill = "#FFA000" if current_fill == "#FFC107" else "#FFC107"
                    app.status_indicator.itemconfig("indicator", fill=new_fill)
                    
                    # Button pulsing effect
                    current_btn_bg = app.start_call_button.cget("bg")
                    new_btn_bg = "#00796B" if current_btn_bg == "#00AD96" else "#00AD96"
                    app.start_call_button.config(bg=new_btn_bg)
                    
                    # Schedule next animation frame
                    app.root.after(200, lambda: pulse_animation(count + 1))
                
                # Start animation
                pulse_animation()
            else:
                # If no status indicator, just wait briefly then proceed
                def delayed_send():
                    # Restore original button color
                    app.start_call_button.config(bg=original_bg)
                    
                    # Send the message to minion_terminal
                    if hasattr(app.__class__, 'minion_terminal'):
                        # Set the text entry in minion_terminal to this message
                        app.__class__.minion_terminal.text_entry.delete("1.0", tk.END)
                        app.__class__.minion_terminal.text_entry.insert("1.0", message)
                        
                        # Start the minion conversation
                        app.__class__.minion_terminal.start_minion_conversation()
                        
                        # Clear the input field after sending
                        app.text_entry.delete("1.0", tk.END)
                        app.text_entry.insert("1.0", "Type a message...")
                        app.text_entry.config(fg="#8696A0")
                        
                        # Update status
                        if hasattr(app, 'status_label'):
                            app.status_label.config(text=f"{MessageHandlers.SEND_EMOJI} Message sent to Minion")
                        
                        # Also add a confirmation in the response area
                        if hasattr(app, 'response_text'):
                            app.response_text.config(state=tk.NORMAL)
                            # Add separator if needed
                            if MessageHandlers.preserve_history and app.response_text.get("1.0", tk.END).strip():
                                app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
                            # Add confirmation message
                            app.response_text.insert(tk.END, f"{MessageHandlers.SEND_EMOJI} Message sent to Minion: \"{message}\"\n", "system_header")
                            app.response_text.config(state=tk.DISABLED)
                            MessageHandlers._ensure_autoscroll(app)
                    else:
                        messagebox.showerror("Error", "Minion terminal not initialized.")
                
                # Delay slightly for visual effect
                app.root.after(800, delayed_send)

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
        """Set up the control buttons for the app."""
        # Create modern control panel
        app.control_frame = tk.Frame(app.main_frame, bg=UIComponents.DARK_BG, height=60)
        app.control_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 10))
        
        # Create a container for buttons with rounded corners
        button_container = tk.Frame(app.control_frame, bg=UIComponents.DARK_BG)
        button_container.pack(pady=(5, 5))
        
        # Call button with icon
        app.call_btn = UIComponents.create_circular_button(
            app, 
            button_container, 
            size=50, 
            color=UIComponents.HIGHLIGHT_COLOR, 
            text=MessageHandlers.CALL_EMOJI,
            command=app.toggle_call
        )
        app.call_btn.pack(side=tk.LEFT, padx=10)
        
        # Send to minion button with icon
        app.send_minion_btn = UIComponents.create_circular_button(
            app, 
            button_container, 
            size=50, 
            color=UIComponents.MESSAGE_BG_SENT, 
            text=MessageHandlers.SEND_EMOJI,
            command=UIComponents.send_to_minion_terminal
        )
        app.send_minion_btn.pack(side=tk.LEFT, padx=10)
        
        # History toggle button with icon
        app.history_btn = UIComponents.create_circular_button(
            app, 
            button_container, 
            size=50, 
            color=UIComponents.MESSAGE_BG_RECEIVED if MessageHandlers.preserve_history else UIComponents.INACTIVE_COLOR, 
            text=MessageHandlers.SAVE_EMOJI,
            command=UIComponents.toggle_history_btn
        )
        app.history_btn.pack(side=tk.LEFT, padx=10)
        
        # Settings button with icon
        app.settings_btn = UIComponents.create_circular_button(
            app, 
            button_container, 
            size=50, 
            color=UIComponents.MESSAGE_BG_RECEIVED, 
            text=MessageHandlers.SETTINGS_ICON,
            command=lambda: UIComponents._show_menu(app)
        )
        app.settings_btn.pack(side=tk.LEFT, padx=10)

    @staticmethod
    def toggle_history_btn(app):
        """Toggle history preservation and update button appearance."""
        preserve = app.toggle_preserve_history()
        if preserve:
            app.history_btn.config(
                text="ON",
                bg="#00BFA5"  # Green color when active
            )
        else:
            app.history_btn.config(
                text="OFF",
                bg="#262D31"  # Default dark color when inactive
            )

    @staticmethod
    def create_circular_button(app, parent, size, color, text, command=None):
        """Create a circular button with the given parameters."""
        # Create a canvas for the circular button
        canvas = tk.Canvas(
            parent, 
            width=size, 
            height=size, 
            bg=UIComponents.DARK_BG, 
            highlightthickness=0
        )
        
        # Draw the circular button background
        canvas.create_oval(
            0, 0, size, size, 
            fill=color, 
            outline=UIComponents.INACTIVE_COLOR, 
            width=1
        )
        
        # Add the text in the center of the button
        canvas.create_text(
            size//2, 
            size//2, 
            text=text, 
            font=("Segoe UI Emoji", size//3), 
            fill=UIComponents.TEXT_COLOR
        )
        
        # Bind button events for interactivity
        def on_enter(e):
            # Highlight button on hover
            canvas.create_oval(
                0, 0, size, size, 
                fill=color, 
                outline=UIComponents.HIGHLIGHT_COLOR, 
                width=2,
                tags="hover"
            )
            canvas.create_text(
                size//2, 
                size//2, 
                text=text, 
                font=("Segoe UI Emoji", size//3), 
                fill=UIComponents.TEXT_COLOR,
                tags="hover_text"
            )
            
        def on_leave(e):
            # Remove highlight on mouse leave
            canvas.delete("hover")
            canvas.delete("hover_text")
            
        def on_click(e):
            # Animate button click
            animate_click()
            if command:
                command(e if command.__code__.co_argcount > 0 else None)
                
        def animate_click():
            original_size = size
            steps = 5
            
            def _animate(step=0):
                if step >= steps * 2:  # Animation completed
                    return
                    
                # Calculate size variation
                if step < steps:  # Shrinking
                    current_size = original_size - (step * 2)
                else:  # Growing
                    current_size = (original_size - (steps * 2)) + ((step - steps) * 2)
                    
                # Clear and redraw
                canvas.delete("animation")
                canvas.create_oval(
                    (size - current_size) // 2, 
                    (size - current_size) // 2, 
                    size - (size - current_size) // 2, 
                    size - (size - current_size) // 2, 
                    fill=color, 
                    outline=UIComponents.HIGHLIGHT_COLOR, 
                    width=2,
                    tags="animation"
                )
                canvas.create_text(
                    size//2, 
                    size//2, 
                    text=text, 
                    font=("Segoe UI Emoji", int(size//3 * (current_size/original_size))), 
                    fill=UIComponents.TEXT_COLOR,
                    tags="animation"
                )
                
                if step < steps * 2 - 1:
                    canvas.after(UIComponents.ANIMATION_SPEED // 2, _animate, step + 1)
                else:
                    canvas.delete("animation")
                    
            _animate()
                
        # Add event bindings
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        canvas.bind("<Button-1>", on_click)
        
        return canvas 