import tkinter as tk
from tkinter import messagebox, PhotoImage

from datetime import datetime
from main import main

import threading
import queue
import sys
import io
from contextlib import redirect_stdout

# Queue for communication between threads
input_queue = queue.Queue()
output_queue = queue.Queue()

class StdoutRedirector(io.StringIO):
    def __init__(self, text_widget, app_instance):
        super().__init__()
        self.text_widget = text_widget
        self.app_instance = app_instance
        
    def write(self, string):
        super().write(string)
        # Schedule the update on the main thread
        self.app_instance.root.after(0, lambda: self.update_text_widget(string))
    
    def update_text_widget(self, string):
        # Enable the text widget for updating
        self.text_widget.config(state=tk.NORMAL)
        
        # Only process messages relevant to this app instance
        app_model_label = self.app_instance.model_label
        
        # Handle thinking messages
        if "★ Worker (Local) is thinking... ★" in string:
            if app_model_label == "Worker (Local)":
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert(tk.END, string, "worker_thinking")
                self.app_instance.is_thinking = True
                self.app_instance.status_label.config(text="Worker is thinking...")
            elif self.app_instance.connected_to and self.app_instance.connected_to.model_label == "Worker (Local)":
                self.app_instance.connected_to.is_thinking = True
                self.app_instance.connected_to.show_thinking_in_response(True)
        elif "★ Supervisor (Remote) is thinking... ★" in string:
            if app_model_label == "Supervisor (Remote)":
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert(tk.END, string, "supervisor_thinking")
                self.app_instance.is_thinking = True
                self.app_instance.status_label.config(text="Supervisor is thinking...")
            elif self.app_instance.connected_to and self.app_instance.connected_to.model_label == "Supervisor (Remote)":
                self.app_instance.connected_to.is_thinking = True
                self.app_instance.connected_to.show_thinking_in_response(True)
        # Handle Worker messages
        elif "@Worker:" in string:
            if app_model_label == "Worker (Local)":
                self.app_instance.is_thinking = False
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert(tk.END, string.replace("@Worker:", "").strip(), "worker_message")
                self.app_instance.status_label.config(text="Displaying response")
                
                # Force exit thinking state to ensure UI is updated
                self.app_instance.root.after(0, lambda: self.app_instance._force_exit_thinking_state())
            elif self.app_instance.connected_to and self.app_instance.connected_to.model_label == "Worker (Local)":
                self.app_instance.connected_to.is_thinking = False
                self.app_instance.connected_to.status_label.config(text="Processing complete")
                
                # Force exit thinking state for connected app too
                self.app_instance.root.after(0, lambda: self.app_instance.connected_to._force_exit_thinking_state())
        # Handle Supervisor messages
        elif "@Supervisor:" in string:
            if app_model_label == "Supervisor (Remote)":
                self.app_instance.is_thinking = False
                self.text_widget.delete("1.0", tk.END)
                self.text_widget.insert(tk.END, string.replace("@Supervisor:", "").strip(), "supervisor_message")
                self.app_instance.status_label.config(text="Displaying response")
                
                # Force exit thinking state to ensure UI is updated
                self.app_instance.root.after(0, lambda: self.app_instance._force_exit_thinking_state())
            elif self.app_instance.connected_to and self.app_instance.connected_to.model_label == "Supervisor (Remote)":
                self.app_instance.connected_to.is_thinking = False
                self.app_instance.connected_to.status_label.config(text="Processing complete")
                
                # Force exit thinking state for connected app too
                self.app_instance.root.after(0, lambda: self.app_instance.connected_to._force_exit_thinking_state())
        else:
            # For regular messages (not thinking status or prefixed messages)
            if string.strip():  # Only process if there's actual content
                if self.app_instance.is_thinking:
                    self.app_instance.is_thinking = False
                    self.text_widget.delete("1.0", tk.END)
                
                if app_model_label == "Worker (Local)":
                    self.text_widget.insert(tk.END, string, "worker_message")
                    self.app_instance.status_label.config(text="Processing complete")
                    # Explicitly force exit thinking state to ensure dialog is shown
                    self.app_instance.root.after(0, lambda: self.app_instance._force_exit_thinking_state())
                elif app_model_label == "Supervisor (Remote)":
                    self.text_widget.insert(tk.END, string, "supervisor_message")
                    self.app_instance.status_label.config(text="Processing complete")
                    # Explicitly force exit thinking state to ensure dialog is shown
                    self.app_instance.root.after(0, lambda: self.app_instance._force_exit_thinking_state())

class VoiceCallApp:
    # Class-level variables to track instances and call state
    instances = []
    active_call = None
    waiting_call = None

    def __init__(self, root, model_name, model_path):
        self.root = root
        self.model_name = model_name
        self.model_path = model_path
        self.model_label = "Model"  # Default label, will be overridden
        self.setup_ui()
        self.call_active = False
        self.call_start_time = None
        self.duration_timer = None
        self.connected_to = None
        self.is_thinking = False
        self._thinking_after_id = None  # Initialize the thinking animation timer ID
        self._pending_replace_id = None  # Track pending replacements
        self._pending_message = None  # Store pending message for delayed display
        self._pending_tag = None  # Store pending tag for delayed display
        
        # Add this instance to the list of instances
        VoiceCallApp.instances.append(self)
        
    def setup_ui(self):
        # Main configuration
        self.root.configure(bg="#121B22")
        self.root.geometry("400x650")
        
        # Main frame
        self.main_frame = tk.Frame(self.root, bg="#121B22")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup UI sections
        self.setup_top_bar()
        self.setup_profile_section()
        self.setup_status_section()
        self.setup_prompt_section()
        self.setup_control_buttons()
        
    def setup_top_bar(self):
        self.top_frame = tk.Frame(self.main_frame, bg="#1F2C34", height=60)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))
        self.secure_label = tk.Label(
            self.top_frame, 
            text="End-to-end encrypted", 
            font=("Segoe UI", 10), 
            fg="#8696A0", 
            bg="#1F2C34"
        )
        self.secure_label.pack(pady=(10, 0))

    def setup_profile_section(self):
        self.profile_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.profile_frame.pack(pady=30)
        
        self.profile_circle = tk.Canvas(
            self.profile_frame, 
            width=150, 
            height=150, 
            bg="#121B22", 
            highlightthickness=0
        )
        self.profile_circle.pack()
        self.profile_circle.create_oval(10, 10, 140, 140, fill="#128C7E", outline="#25D366", width=2)
        self.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
        self.name_label = tk.Label(
            self.profile_frame, 
            text=self.model_label, 
            font=("Segoe UI", 24, "bold"), 
            fg="white", 
            bg="#121B22"
        )
        self.name_label.pack(pady=(15, 5))
        
        self.model_info_label = tk.Label(
            self.profile_frame, 
            text=self.model_name, 
            font=("Segoe UI", 14), 
            fg="#00BFA5", 
            bg="#121B22"
        )
        self.model_info_label.pack()

    def setup_status_section(self):
        self.status_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            self.status_frame, 
            font=("Segoe UI", 14), 
            fg="#8696A0", 
            bg="#121B22"
        )
        self.status_label.pack()
        
        self.duration_label = tk.Label(
            self.status_frame, 
            text="", 
            font=("Segoe UI", 14), 
            fg="#8696A0", 
            bg="#121B22"
        )
        
        self.start_call_button = tk.Button(
            self.status_frame, 
            text="Start Calling", 
            font=("Segoe UI", 12), 
            bg="#00BFA5", 
            fg="white",
            command=self.toggle_call, 
            padx=10, 
            pady=5, 
            relief=tk.RAISED, 
            bd=0
        )
        self.start_call_button.pack(pady=10)
        
        self.stop_call_button = tk.Button(
            self.status_frame, 
            text="Stop Calling", 
            font=("Segoe UI", 12), 
            bg="#FF0000", 
            fg="white",
            command=self.end_call, 
            padx=10, 
            pady=5, 
            relief=tk.RAISED, 
            bd=0
        )

    def setup_prompt_section(self):
        self.prompt_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Response area (where conversation appears)
        self.response_frame = tk.Frame(self.prompt_frame, bg="#121B22")
        self.response_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label for the response area
        self.response_label = tk.Label(
            self.response_frame, 
            text="Conversation", 
            font=("Segoe UI", 12, "bold"), 
            fg="white", 
            bg="#121B22", 
            anchor="w"
        )
        self.response_label.pack(fill=tk.X, pady=(0, 5))
        
        # Response text widget with scrollbar
        response_scroll = tk.Scrollbar(self.response_frame)
        response_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.response_text = tk.Text(
            self.response_frame, 
            wrap=tk.WORD, 
            bg="#1F2C34", 
            fg="white", 
            insertbackground="white", 
            height=10, 
            font=("Segoe UI", 11), 
            relief=tk.FLAT, 
            padx=10, 
            pady=10
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
        self.response_text.config(yscrollcommand=response_scroll.set)
        response_scroll.config(command=self.response_text.yview)
        
        # Configure text tags for different message types
        self.response_text.tag_configure("worker_message", foreground="#DCF8C6", lmargin1=20, lmargin2=20)
        self.response_text.tag_configure("supervisor_message", foreground="#E2F7CB", lmargin1=20, lmargin2=20)
        self.response_text.tag_configure("worker_thinking", foreground="#8696A0", font=("Segoe UI", 11, "italic"))
        self.response_text.tag_configure("supervisor_thinking", foreground="#8696A0", font=("Segoe UI", 11, "italic"))
        self.response_text.tag_configure("worker_question_header", foreground="#25D366", font=("Segoe UI", 11, "bold"))
        self.response_text.tag_configure("supervisor_question_header", foreground="#34B7F1", font=("Segoe UI", 11, "bold"))
        self.response_text.tag_configure("worker_answer_header", foreground="#25D366", font=("Segoe UI", 11, "bold"))
        self.response_text.tag_configure("supervisor_answer_header", foreground="#34B7F1", font=("Segoe UI", 11, "bold"))
        self.response_text.tag_configure("waiting_message", foreground="#8696A0", font=("Segoe UI", 11, "italic"))
        
        # Set initial readonly state
        self.response_text.config(state=tk.DISABLED)
        
        # Text entry area at the bottom
        self.entry_frame = tk.Frame(self.prompt_frame, bg="#1F2C34", height=100)
        self.entry_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Add a label for the text entry area
        self.prompt_label = tk.Label(
            self.prompt_frame, 
            text="Your message:", 
            font=("Segoe UI", 12), 
            fg="#8696A0", 
            bg="#121B22", 
            anchor="w"
        )
        self.prompt_label.pack(fill=tk.X, pady=(10, 5))
        
        self.text_entry = tk.Text(
            self.entry_frame, 
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
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        # Bind Return key to toggle_call
        self.text_entry.bind("<Return>", self.toggle_call)

    def setup_control_buttons(self):
        self.controls_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        self.buttons_frame = tk.Frame(self.controls_frame, bg="#121B22")
        self.buttons_frame.pack(pady=20)
        
        button_size = 60
        self.mute_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Mute")
        self.mute_frame.pack(side=tk.LEFT, padx=15)
        
        self.call_button_frame = self.create_circular_button(
            self.buttons_frame, 
            button_size + 20, 
            "#00BFA5", 
            "Call", 
            command=self.toggle_call
        )
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
        
        self.speaker_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Speaker")
        self.speaker_frame.pack(side=tk.LEFT, padx=15)

    def create_circular_button(self, parent, size, color, text, command=None):
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

    def toggle_call(self):
        if not self.call_active:
            self.start_call()
        else:
            self.end_call()

    def start_call(self):
        """Start a call between the current app and another VoiceCallApp instance."""
        # Get the input text from the text entry
        input_text = self.text_entry.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Input Required", "Please enter some text before calling.")
            return
        
        # Clear the input area
        self.text_entry.delete("1.0", tk.END)
        
        # Show thinking state in both apps
        self.show_thinking_in_response(True)
        if self.connected_to:
            self.connected_to.show_thinking_in_response(True)
        
        # Run main() in a separate thread to avoid blocking the UI
        def run_main_in_thread():
            try:
                # Custom redirector for UI updates
                class DirectUIUpdatingRedirector(StdoutRedirector):
                    def update_text_widget(self, string):
                        # Helper function to check if a widget still exists
                        def widget_exists(widget):
                            try:
                                widget.winfo_exists()
                                return True
                            except:
                                return False
                        
                        # Process completion messages
                        if "@Worker: Processing complete" in string:
                            worker_app = None
                            if self.app_instance.model_label == "Worker (Local)":
                                worker_app = self.app_instance
                            elif (self.app_instance.connected_to and 
                                  self.app_instance.connected_to.model_label == "Worker (Local)"):
                                worker_app = self.app_instance.connected_to
                            
                            if worker_app and widget_exists(worker_app.root):
                                # Only force exit the thinking state, don't clear any displayed content immediately
                                worker_app.root.after(0, lambda: self._update_thinking_state_only(worker_app, False))
                            return
                        
                        elif "@Supervisor: Processing complete" in string:
                            supervisor_app = None
                            if self.app_instance.model_label == "Supervisor (Remote)":
                                supervisor_app = self.app_instance
                            elif (self.app_instance.connected_to and 
                                  self.app_instance.connected_to.model_label == "Supervisor (Remote)"):
                                supervisor_app = self.app_instance.connected_to
                            
                            if supervisor_app and widget_exists(supervisor_app.root):
                                # Only force exit the thinking state, don't clear any displayed content immediately
                                supervisor_app.root.after(0, lambda: self._update_thinking_state_only(supervisor_app, False))
                            return
                        
                        # Process Worker messages
                        elif "@Worker:" in string:
                            # Find the Worker and Supervisor apps
                            worker_app = None
                            supervisor_app = None
                            
                            if self.app_instance.model_label == "Worker (Local)":
                                worker_app = self.app_instance
                                supervisor_app = self.app_instance.connected_to
                            elif self.app_instance.model_label == "Supervisor (Remote)":
                                supervisor_app = self.app_instance
                                worker_app = self.app_instance.connected_to
                            
                            # Process the message content
                            display_text = string.replace("@Worker:", "").strip()
                            is_question = display_text.endswith("?") or "?" in display_text
                            
                            # Update Worker app - ensure thinking state is exited first
                            if worker_app and widget_exists(worker_app.root):
                                # First, update the thinking state only (don't clear content)
                                worker_app.root.after(100, lambda w=worker_app: 
                                    self.safe_call(w, lambda: self._update_thinking_state_only(w, False)))
                                # Then update with the new response text after 5 seconds if there was content
                                worker_app.root.after(5000, lambda w=worker_app, t=display_text: 
                                    self.safe_call(w, lambda: w._update_response_text(t, "worker_message")))
                                
                                if is_question and supervisor_app and widget_exists(supervisor_app.root):
                                    worker_app.root.after(5500, lambda w=worker_app, 
                                        t=f"{display_text}\n\nWaiting for Supervisor to respond...": 
                                        self.safe_call(w, lambda: w._update_response_text(t, "worker_message")))
                                    worker_app.root.after(6000, lambda s=supervisor_app: 
                                        self.safe_call(s, lambda: s.show_thinking_in_response(True)))
                            
                            # Update Supervisor app
                            if supervisor_app and widget_exists(supervisor_app.root):
                                # First, update the thinking state only (don't clear content)
                                supervisor_app.root.after(100, lambda s=supervisor_app: 
                                    self.safe_call(s, lambda: self._update_thinking_state_only(s, False)))
                                # Then update with the new response text after 5 seconds if there was content
                                supervisor_app.root.after(5000, lambda s=supervisor_app, t=f"Worker: {display_text}": 
                                    self.safe_call(s, lambda: s._update_response_text(t, "worker_message")))
                                
                                if is_question:
                                    supervisor_app.root.after(5500, lambda s=supervisor_app: 
                                        self.safe_call(s, lambda: s.show_thinking_in_response(True)))
                            
                            return
                        
                        # Process Supervisor messages
                        elif "@Supervisor:" in string:
                            # Find the Worker and Supervisor apps
                            worker_app = None
                            supervisor_app = None
                            
                            if self.app_instance.model_label == "Worker (Local)":
                                worker_app = self.app_instance
                                supervisor_app = self.app_instance.connected_to
                            elif self.app_instance.model_label == "Supervisor (Remote)":
                                supervisor_app = self.app_instance
                                worker_app = self.app_instance.connected_to
                            
                            # Process the message content
                            display_text = string.replace("@Supervisor:", "").strip()
                            is_question = display_text.endswith("?") or "?" in display_text
                            
                            # Update Supervisor app
                            if supervisor_app and widget_exists(supervisor_app.root):
                                # First, update the thinking state only (don't clear content)
                                supervisor_app.root.after(100, lambda s=supervisor_app: 
                                    self.safe_call(s, lambda: self._update_thinking_state_only(s, False)))
                                # Then update with the new response text after 5 seconds if there was content
                                supervisor_app.root.after(5000, lambda s=supervisor_app, t=display_text: 
                                    self.safe_call(s, lambda: s._update_response_text(t, "supervisor_message")))
                                
                                if is_question and worker_app and widget_exists(worker_app.root):
                                    supervisor_app.root.after(5500, lambda s=supervisor_app, 
                                        t=f"{display_text}\n\nWaiting for Worker to respond...": 
                                        self.safe_call(s, lambda: s._update_response_text(t, "supervisor_message")))
                                    supervisor_app.root.after(6000, lambda w=worker_app: 
                                        self.safe_call(w, lambda: w.show_thinking_in_response(True)))
                            
                            # Update Worker app
                            if worker_app and widget_exists(worker_app.root):
                                # First, update the thinking state only (don't clear content)
                                worker_app.root.after(100, lambda w=worker_app: 
                                    self.safe_call(w, lambda: self._update_thinking_state_only(w, False)))
                                # Then update with the new response text after 5 seconds if there was content
                                worker_app.root.after(5000, lambda w=worker_app, t=f"Supervisor: {display_text}": 
                                    self.safe_call(w, lambda: w._update_response_text(t, "supervisor_message")))
                                
                                if is_question:
                                    worker_app.root.after(5500, lambda w=worker_app: 
                                        self.safe_call(w, lambda: w.show_thinking_in_response(True)))
                            
                            return
                        
                        # Continue with normal processing for other messages
                        super().update_text_widget(string)
                    
                    def safe_call(self, app, callback):
                        """Safely call a function on an app instance, checking if the app is still valid."""
                        try:
                            if app and app.root.winfo_exists():
                                callback()
                        except Exception:
                            # Silently fail if the widget no longer exists
                            pass
                            
                    def _update_thinking_state_only(self, app, is_thinking):
                        """Update only the thinking state flag without clearing dialog content."""
                        try:
                            if app and app.root.winfo_exists():
                                # Set the thinking state
                                app.is_thinking = is_thinking
                                
                                # Update the status label
                                if is_thinking:
                                    if app.model_label == "Worker (Local)":
                                        app.status_label.config(text="Worker is thinking...")
                                    elif app.model_label == "Supervisor (Remote)":
                                        app.status_label.config(text="Supervisor is thinking...")
                                else:
                                    if app.model_label == "Worker (Local)":
                                        app.status_label.config(text="Worker ready")
                                    elif app.model_label == "Supervisor (Remote)":
                                        app.status_label.config(text="Supervisor ready")
                                
                                # Cancel any thinking animation
                                if app._thinking_after_id:
                                    try:
                                        app.root.after_cancel(app._thinking_after_id)
                                        app._thinking_after_id = None
                                    except:
                                        pass
                        except:
                            pass
                
                # Use the custom redirector
                stdout_redirector = DirectUIUpdatingRedirector(self.response_text, self)
                with redirect_stdout(stdout_redirector):
                    # Process the input in the background thread
                    main(input_text)
                
                # Always force both apps to exit thinking state after processing
                self.root.after(0, lambda: self._force_exit_thinking_state())
                if self.connected_to:
                    self.connected_to.root.after(0, lambda: self.connected_to._force_exit_thinking_state())
            
            except Exception as e:
                # Handle errors and update UI from the main thread
                error_message = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {error_message}"))
                self.root.after(0, lambda: self.append_to_response(f"Error: {error_message}"))
                self.root.after(0, lambda: self._force_exit_thinking_state())
                if self.connected_to:
                    self.connected_to.root.after(0, lambda: self.connected_to._force_exit_thinking_state())

        # Create and start the thread
        threading_thread = threading.Thread(target=run_main_in_thread)
        threading_thread.daemon = True  # Thread will exit when main program exits
        threading_thread.start()
        
        # Keep the UI responsive by processing events
        self.root.update_idletasks()
        
        # Check if there's already an active call
        if (VoiceCallApp.active_call is not None and 
            VoiceCallApp.active_call != self and 
            VoiceCallApp.active_call != self.connected_to):
            messagebox.showerror("Error", "Another call is already in progress!")
            return
            
        # Set this instance as the waiting call if no other call is waiting
        if VoiceCallApp.waiting_call is None:
            VoiceCallApp.waiting_call = self
            self.status_label.config(text="Waiting for other party to accept...")
            self.call_active = True
            self.start_call_button.pack_forget()
            self.stop_call_button.pack(pady=10)
            self.text_entry.config(state=tk.DISABLED)
            
            # Find another available instance and automatically connect
            for app in VoiceCallApp.instances:
                if app != self and not app.call_active:
                    # Auto-accept call on the other instance
                    app.text_entry.insert("1.0", f"Auto-accepted call {app.model_path}\n")
                    app.connected_to = self
                    self.connected_to = app
                    VoiceCallApp.active_call = (self, app)
                    
                    # Start the call for both apps
                    self._establish_call(app)
                    app._establish_call(self)
                    
                    VoiceCallApp.waiting_call = None
        else:
            # Connect the calls if there's a waiting call
            other_app = VoiceCallApp.waiting_call
            VoiceCallApp.active_call = (other_app, self)
            
            # Setup both apps for the call
            self.connected_to = other_app
            other_app.connected_to = self
            
            # Start the call for both apps
            self._establish_call(other_app)
            other_app._establish_call(self)
            
            VoiceCallApp.waiting_call = None

    def _establish_call(self, other_app):
        self.call_active = True
        self.call_start_time = datetime.now()
        self.status_label.config(text=f"Connected to {other_app.model_name}")
        self.duration_label.pack()
        self.start_call_button.pack_forget()
        self.stop_call_button.pack(pady=10)
        self.text_entry.config(state=tk.DISABLED)
        
        # Clear any previous response first
        self.clear_response()
        
        # Show appropriate thinking status
        self.show_thinking_in_response(True)
        
        # Start the duration timer
        self.update_duration()
        
        # Change call button to red for ending the call
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(
            self.buttons_frame, 
            80, 
            "#FF0000", 
            "End", 
            command=self.toggle_call
        )
        self.call_button_frame.pack(side=tk.LEFT, padx=15)

    def end_call(self):
        """End the active call."""
        if not self.call_active:
            return
            
        # Update UI state
        self.call_active = False
        self.call_start_time = None
        self.stop_call_button.pack_forget()
        self.start_call_button.pack(pady=10)
        
        # Reset the duration label if it's being displayed
        if self.duration_label.winfo_master():
            self.duration_label.pack_forget()
            
        # Cancel any pending after IDs for animations
        if self._thinking_after_id:
            try:
                self.root.after_cancel(self._thinking_after_id)
            except:
                pass
            self._thinking_after_id = None
        
        # Cancel any pending replacements
        if self._pending_replace_id:
            try:
                self.root.after_cancel(self._pending_replace_id)
            except:
                pass
            self._pending_replace_id = None
            self._pending_message = None
            self._pending_tag = None
            
        # Cancel any other pending after callbacks
        try:
            for after_id in self.root.tk.call('after', 'info'):
                try:
                    self.root.after_cancel(after_id)
                except:
                    pass
        except:
            pass
            
        # Terminate the call relationship
        self._terminate_call()
        
        # Reset response areas
        self.status_label.config(text="Call ended")
        
        # Reset thinking state
        self.is_thinking = False
        
        # Clear the response area and show the call ended message
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.insert(tk.END, "Call ended. Enter a new message and press Call to start a new conversation.", "supervisor_message")
        self.response_text.config(state=tk.DISABLED)

    def _terminate_call(self):
        text_input = self.text_entry.get("1.0", tk.END)
        messagebox.showinfo("Call Complete", f"Voice call ended!\nMessage: {text_input}")
        
        self.call_active = False
        self.call_start_time = None
        
        # Clear status and thinking indicators
        self.status_label.config(text="")
        self.duration_label.pack_forget()
        self.start_call_button.pack(pady=10)
        self.stop_call_button.pack_forget()
        
        # Clear the text entry and response areas
        self.text_entry.config(state=tk.NORMAL)
        self.text_entry.delete("1.0", tk.END)
        self.clear_response()
        
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#00BFA5", "Call", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)

    def update_duration(self):
        if self.call_active and self.call_start_time:
            now = datetime.now()
            diff = now - self.call_start_time
            minutes = diff.seconds // 60
            seconds = diff.seconds % 60
            call_duration = f"{minutes:02d}:{seconds:02d}"
            self.duration_label.config(text=call_duration)
            self.root.after(1000, self.update_duration)

    def update_thinking_status(self):
        """Update the status label to show the thinking message with stars."""
        try:
            # Check if widgets still exist
            if not (hasattr(self, 'response_text') and self.response_text.winfo_exists() and
                    hasattr(self, 'status_label') and self.status_label.winfo_exists()):
                # If widgets don't exist, don't schedule any more updates
                self.is_thinking = False
                return
                
            # Only show thinking status relevant to this app instance
            if self.model_label == "Supervisor (Remote)":
                thinking_message = "★ Supervisor (Remote) is thinking... ★"
                # Update the status label
                self.status_label.config(text=thinking_message)
                
                # Also update the response text widget with the thinking status
                self.response_text.config(state=tk.NORMAL)
                self.response_text.delete("1.0", tk.END)  # Clear any previous content
                self.response_text.insert(tk.END, thinking_message + "\n", "supervisor_thinking")
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            elif self.model_label == "Worker (Local)":
                thinking_message = "★ Worker (Local) is thinking... ★"
                # Update the status label
                self.status_label.config(text=thinking_message)
                
                # Also update the response text widget with the thinking status
                self.response_text.config(state=tk.NORMAL)
                self.response_text.delete("1.0", tk.END)  # Clear any previous content
                self.response_text.insert(tk.END, thinking_message + "\n", "worker_thinking")
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
            else:
                # Generic handling for any other model labels
                thinking_message = f"★ {self.model_label} is thinking... ★"
                self.status_label.config(text=thinking_message)
                
            # Keep the UI responsive
            try:
                self.root.update_idletasks()
            except:
                # If update_idletasks fails, the window might be gone
                self.is_thinking = False
                return
                
            # Schedule the next update if still in thinking state
            if self.is_thinking and self.root.winfo_exists():
                # Cancel any existing timer to avoid multiple timers
                if self._thinking_after_id:
                    try:
                        self.root.after_cancel(self._thinking_after_id)
                    except:
                        pass
                # Schedule next update in 500ms
                self._thinking_after_id = self.root.after(500, self.update_thinking_status)
            else:
                # Clear thinking state if the window is gone
                self.is_thinking = False
                self._thinking_after_id = None
        except (tk.TclError, RuntimeError, Exception) as e:
            # If any Tkinter error occurs, stop the thinking animation
            self.is_thinking = False
            self._thinking_after_id = None
            # print(f"Error in thinking animation: {e}")

    def clear_response(self):
        """Clear the response text area."""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete("1.0", tk.END)
        self.response_text.config(state=tk.DISABLED)
    
    def append_to_response(self, text):
        """Append text to the response area."""
        # Check if this is a thinking status message that should be filtered
        if "★ Worker (Local) is thinking... ★" in text and self.model_label != "Worker (Local)":
            # Don't show Worker thinking message in Supervisor app
            return
        elif "★ Supervisor (Remote) is thinking... ★" in text and self.model_label != "Supervisor (Remote)":
            # Don't show Supervisor thinking message in Worker app
            return
        
        # Filter responses to ensure they only appear in the correct app
        if "@Worker:" in text and self.model_label != "Worker (Local)":
            # Don't show Worker responses in Supervisor app
            return
        elif "@Supervisor:" in text and self.model_label != "Supervisor (Remote)":
            # Don't show Supervisor responses in Worker app
            return
        
        # Special handling for thinking messages
        if ("★ Worker (Local) is thinking... ★" in text and self.model_label == "Worker (Local)") or \
           ("★ Supervisor (Remote) is thinking... ★" in text and self.model_label == "Supervisor (Remote)"):
            # For thinking messages, clear the response area first to show only the thinking status
            self.response_text.config(state=tk.NORMAL)
            self.response_text.delete("1.0", tk.END)
            
            # Apply appropriate formatting based on message content
            if "★ Worker (Local) is thinking... ★" in text:
                self.response_text.insert(tk.END, text + "\n", "worker_thinking")
                self.is_thinking = True
            elif "★ Supervisor (Remote) is thinking... ★" in text:
                self.response_text.insert(tk.END, text + "\n", "supervisor_thinking")
                self.is_thinking = True
                
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
            
            # Also update the status label
            self.status_label.config(text=text)
            return
        
        # Handle explicit "Processing complete" messages
        if "@Worker: Processing complete" in text and self.model_label == "Worker (Local)":
            self.is_thinking = False
            self.status_label.config(text="Processing complete")
            if self.connected_to:
                # Just update the status in the connected app, don't display the message
                self.connected_to.root.after(0, lambda: self.connected_to.status_label.config(text="Processing complete"))
            return  # Don't actually display this message
        
        if "@Supervisor: Processing complete" in text and self.model_label == "Supervisor (Remote)":
            self.is_thinking = False
            self.status_label.config(text="Processing complete")
            if self.connected_to:
                # Just update the status in the connected app, don't display the message
                self.connected_to.root.after(0, lambda: self.connected_to.status_label.config(text="Processing complete"))
            return  # Don't actually display this message
            
        # For all other messages, or if the thinking message matches this app's identity
        self.response_text.config(state=tk.NORMAL)
        
        # First, check if we need to clear thinking status
        if self.is_thinking:
            # Reset thinking state and clear existing content
            self.is_thinking = False
            # Clear existing content if it's a thinking message
            self.response_text.delete("1.0", tk.END)
            # Update status to normal
            self.status_label.config(text="Processing complete")
        
        # Apply appropriate formatting based on message content
        if "@Worker:" in text:
            # In Worker app, display without the prefix
            if self.model_label == "Worker (Local)":
                display_text = text.replace("@Worker:", "").strip()
                self.response_text.insert(tk.END, display_text + "\n", "worker_message")
            else:
                # In other apps, show with the prefix
                self.response_text.insert(tk.END, text + "\n", "worker_message")
        elif "@Supervisor:" in text:
            # In Supervisor app, display without the prefix
            if self.model_label == "Supervisor (Remote)":
                display_text = text.replace("@Supervisor:", "").strip()
                self.response_text.insert(tk.END, display_text + "\n", "supervisor_message")
            else:
                # In other apps, show with the prefix
                self.response_text.insert(tk.END, text + "\n", "supervisor_message")
        else:
            # For regular messages, or default content from the active model
            if self.model_label == "Worker (Local)" and self.is_thinking == False:
                # If in Worker app and not thinking, treat unlabeled content as worker content
                self.response_text.insert(tk.END, text + "\n", "worker_message")
            elif self.model_label == "Supervisor (Remote)" and self.is_thinking == False:
                # If in Supervisor app and not thinking, treat unlabeled content as supervisor content
                self.response_text.insert(tk.END, text + "\n", "supervisor_message")
            else:
                # Default handling
                self.response_text.insert(tk.END, text + "\n")
            
        self.response_text.see(tk.END)
        self.response_text.config(state=tk.DISABLED)
        
    def show_thinking_in_response(self, is_thinking=True):
        """Shows or hides the thinking indicator in the response area."""
        try:
            # Check if all required widgets exist
            if not (self.response_text.winfo_exists() and self.status_label.winfo_exists()):
                return
                
            if is_thinking:
                # Cancel any existing thinking animation first
                if self._thinking_after_id:
                    try:
                        self.root.after_cancel(self._thinking_after_id)
                    except:
                        pass
                    self._thinking_after_id = None
                
                # Cancel any pending replacement before showing thinking state
                if self._pending_replace_id:
                    try:
                        self.root.after_cancel(self._pending_replace_id)
                        self._pending_replace_id = None
                        self._pending_message = None
                        self._pending_tag = None
                    except:
                        pass
                
                # Update the app's thinking state
                self.is_thinking = is_thinking
                
                # Get current content to check if we should preserve conversation history
                current_text = self.response_text.get("1.0", tk.END).strip()
                preserve_history = (current_text and 
                                   "is thinking" not in current_text and 
                                   "ready for new input" not in current_text)
                
                # Make text widget editable
                self.response_text.config(state=tk.NORMAL)
                
                # If we should preserve history, append thinking indicator
                if preserve_history:
                    self.response_text.insert(tk.END, "\n\n", "")
                    
                    # Add thinking text with appropriate format based on app type
                    if self.model_label == "Worker (Local)":
                        self.response_text.insert(tk.END, "★ Worker (Local) is thinking... ★", "worker_thinking")
                        self.status_label.config(text="Worker is thinking...")
                    elif self.model_label == "Supervisor (Remote)":
                        self.response_text.insert(tk.END, "★ Supervisor (Remote) is thinking... ★", "supervisor_thinking")
                        self.status_label.config(text="Supervisor is thinking...")
                else:
                    # Instead of immediately clearing, wait 5 seconds if there's content worth preserving
                    if current_text and not "is thinking" in current_text and not self.is_thinking:
                        # Schedule thinking indicator to appear after 5 seconds
                        self._pending_message = "thinking"
                        self._pending_tag = "thinking"
                        self._pending_replace_id = self.root.after(5000, lambda: self._delayed_thinking_update())
                        return
                    else:
                        # Clear immediately if the current content isn't worth preserving
                        self.response_text.delete("1.0", tk.END)
                        
                        # Add thinking text with appropriate format based on app type
                        if self.model_label == "Worker (Local)":
                            self.response_text.insert(tk.END, "★ Worker (Local) is thinking... ★", "worker_thinking")
                            self.status_label.config(text="Worker is thinking...")
                        elif self.model_label == "Supervisor (Remote)":
                            self.response_text.insert(tk.END, "★ Supervisor (Remote) is thinking... ★", "supervisor_thinking")
                            self.status_label.config(text="Supervisor is thinking...")
                
                # Start the thinking animation
                self.update_thinking_status()
                
                # Make sure the text is visible and readonly
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
                
                # Ensure thinking state exits after a timeout (safety mechanism)
                # This prevents the app from getting stuck in thinking state
                self.root.after(30000, lambda: self._force_exit_thinking_state_if_still_thinking())
            else:
                # Use the force exit method for consistency
                self._force_exit_thinking_state()
        except (tk.TclError, RuntimeError, Exception) as e:
            # If any Tkinter error occurs, silently fail
            # print(f"Error showing thinking status: {e}")
            pass
            
    def _delayed_thinking_update(self):
        """Handle delayed update for thinking state."""
        try:
            if not self.response_text.winfo_exists():
                return
                
            # Only proceed if we're still in thinking state
            if self.is_thinking:
                self.response_text.config(state=tk.NORMAL)
                self.response_text.delete("1.0", tk.END)
                
                # Add thinking text with appropriate format based on app type
                if self.model_label == "Worker (Local)":
                    self.response_text.insert(tk.END, "★ Worker (Local) is thinking... ★", "worker_thinking")
                    self.status_label.config(text="Worker is thinking...")
                elif self.model_label == "Supervisor (Remote)":
                    self.response_text.insert(tk.END, "★ Supervisor (Remote) is thinking... ★", "supervisor_thinking")
                    self.status_label.config(text="Supervisor is thinking...")
                
                self.response_text.see(tk.END)
                self.response_text.config(state=tk.DISABLED)
                
                # Reset pending status
                self._pending_replace_id = None
                self._pending_message = None
                self._pending_tag = None
        except:
            pass

    def _force_exit_thinking_state_if_still_thinking(self):
        """Force exit thinking state only if app is still in thinking state."""
        if self.is_thinking:
            self._force_exit_thinking_state()

    def _force_exit_thinking_state(self):
        """Forces the app to exit thinking state completely."""
        try:
            # Check if all required widgets exist
            if not (hasattr(self, 'response_text') and self.response_text.winfo_exists() and 
                    hasattr(self, 'status_label') and self.status_label.winfo_exists()):
                return
            
            # Stop any thinking animation
            if self._thinking_after_id:
                try:
                    self.root.after_cancel(self._thinking_after_id)
                except:
                    pass
                self._thinking_after_id = None
            
            # Cancel any pending replacement
            if self._pending_replace_id:
                try:
                    self.root.after_cancel(self._pending_replace_id)
                except:
                    pass
                self._pending_replace_id = None
                self._pending_message = None
                self._pending_tag = None
                
            # Update UI state - important to set this first before manipulating the text
            self.is_thinking = False
            
            # Get the current response text content
            self.response_text.config(state=tk.NORMAL)
            current_text = self.response_text.get("1.0", tk.END).strip()
            
            # Only modify text if it's currently showing a thinking indicator
            if "is thinking" in current_text:
                # Clear the thinking indicator text and prepare for regular messages
                self.response_text.delete("1.0", tk.END)
                
                # Only show "ready" message if we're not expecting to immediately show a response
                if self.model_label == "Worker (Local)":
                    self.status_label.config(text="Worker ready")
                elif self.model_label == "Supervisor (Remote)":
                    self.status_label.config(text="Supervisor ready")
            # Otherwise keep the existing response
            else:
                # Just update the status without changing the response text
                if self.model_label == "Worker (Local)":
                    self.status_label.config(text="Worker")
                elif self.model_label == "Supervisor (Remote)":
                    self.status_label.config(text="Supervisor")
                
            # Make sure the text is visible and readonly
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
        except (tk.TclError, RuntimeError, Exception) as e:
            # If any Tkinter error occurs, silently fail
            # print(f"Error exiting thinking state: {e}")
            pass

    def _update_response_text(self, text, tag):
        """Update the response text with the provided text and tag.
        
        This is a helper method used to safely update the response text from UI thread callbacks.
        """
        try:
            # Check if the response_text widget still exists
            if not self.response_text.winfo_exists():
                return
            
            # Cancel any pending replacement
            if self._pending_replace_id:
                try:
                    self.root.after_cancel(self._pending_replace_id)
                    self._pending_replace_id = None
                except:
                    pass
                
            # Ensure the response area is editable
            self.response_text.config(state=tk.NORMAL)
            
            # Instead of clearing, check if there's existing content to append to
            current_text = self.response_text.get("1.0", tk.END).strip()
            
            # If there's valuable conversation content and not just thinking indicators,
            # delay clearing for 5 seconds to let user read it
            if (current_text and 
                not "is thinking" in current_text and 
                not "ready for new input" in current_text and
                not self.is_thinking):
                # Store the pending message
                self._pending_message = text
                self._pending_tag = tag
                # Schedule the update after 5 seconds
                self._pending_replace_id = self.root.after(5000, lambda: self._delayed_text_update(text, tag))
                # Make the text widget read-only again and exit
                self.response_text.config(state=tk.DISABLED)
                return
            
            # Clear thinking indicators or empty ready messages right away
            if not current_text or "is thinking" in current_text or "ready for new input" in current_text:
                # Clear the text area completely first
                self.response_text.delete("1.0", tk.END)
                
                # Check if this is a question and format accordingly
                if text.endswith("?") or "?" in text and not "Waiting for" in text:
                    if tag == "worker_message":
                        self.response_text.insert(tk.END, "Question from Worker:\n", "worker_question_header")
                        self.response_text.insert(tk.END, text, tag)
                    elif tag == "supervisor_message":
                        self.response_text.insert(tk.END, "Question from Supervisor:\n", "supervisor_question_header") 
                        self.response_text.insert(tk.END, text, tag)
                    else:
                        self.response_text.insert(tk.END, text, tag)
                else:
                    if "Waiting for" in text:
                        # For waiting messages, just show the full text
                        self.response_text.insert(tk.END, text, tag)
                    else:
                        # For answers or statements
                        if tag == "worker_message" and not text.startswith("Worker:"):
                            self.response_text.insert(tk.END, "Worker responds:\n", "worker_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        elif tag == "supervisor_message" and not text.startswith("Supervisor:"):
                            self.response_text.insert(tk.END, "Supervisor responds:\n", "supervisor_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        else:
                            self.response_text.insert(tk.END, text, tag)
            else:
                # Append the new message with clear separation
                self.response_text.insert(tk.END, "\n\n", "")
                
                # Check if this is a question and format accordingly
                if text.endswith("?") or "?" in text and not "Waiting for" in text:
                    if tag == "worker_message":
                        self.response_text.insert(tk.END, "Question from Worker:\n", "worker_question_header")
                        self.response_text.insert(tk.END, text, tag)
                    elif tag == "supervisor_message":
                        self.response_text.insert(tk.END, "Question from Supervisor:\n", "supervisor_question_header") 
                        self.response_text.insert(tk.END, text, tag)
                    else:
                        self.response_text.insert(tk.END, text, tag)
                else:
                    if "Waiting for" in text:
                        # For waiting messages, just append the waiting notification
                        parts = text.split("\n\nWaiting for")
                        if len(parts) > 1:
                            self.response_text.insert(tk.END, "\nWaiting for" + parts[1], "waiting_message")
                        else:
                            self.response_text.insert(tk.END, text, tag)
                    else:
                        # For answers or statements
                        if tag == "worker_message" and not text.startswith("Worker:"):
                            self.response_text.insert(tk.END, "Worker responds:\n", "worker_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        elif tag == "supervisor_message" and not text.startswith("Supervisor:"):
                            self.response_text.insert(tk.END, "Supervisor responds:\n", "supervisor_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        else:
                            self.response_text.insert(tk.END, text, tag)
            
            # Make sure the new text is visible
            self.response_text.see(tk.END)
            
            # Reset to read-only
            self.response_text.config(state=tk.DISABLED)
            
            # Update the status label based on who's speaking
            if self.status_label.winfo_exists():
                if tag == "worker_message":
                    is_question = text.endswith("?") or "?" in text and not "Waiting for" in text
                    if is_question:
                        self.status_label.config(text="Worker asking")
                    else:
                        self.status_label.config(text="Worker speaking")
                elif tag == "supervisor_message":
                    is_question = text.endswith("?") or "?" in text and not "Waiting for" in text
                    if is_question:
                        self.status_label.config(text="Supervisor asking")
                    else:
                        self.status_label.config(text="Supervisor speaking")
                else:
                    self.status_label.config(text="Conversation active")
                    
        except (tk.TclError, RuntimeError, Exception) as e:
            # If any Tkinter error occurs, silently fail
            # print(f"Error updating response text: {e}")
            pass
            
    def _delayed_text_update(self, text, tag):
        """Handle delayed update of text."""
        try:
            if not self.response_text.winfo_exists():
                return
                
            # Only proceed if this is still the pending message
            if text == self._pending_message and tag == self._pending_tag:
                # Ensure the response area is editable
                self.response_text.config(state=tk.NORMAL)
                
                # Clear the text area
                self.response_text.delete("1.0", tk.END)
                
                # Check if this is a question and format accordingly
                if text.endswith("?") or "?" in text and not "Waiting for" in text:
                    if tag == "worker_message":
                        self.response_text.insert(tk.END, "Question from Worker:\n", "worker_question_header")
                        self.response_text.insert(tk.END, text, tag)
                    elif tag == "supervisor_message":
                        self.response_text.insert(tk.END, "Question from Supervisor:\n", "supervisor_question_header") 
                        self.response_text.insert(tk.END, text, tag)
                    else:
                        self.response_text.insert(tk.END, text, tag)
                else:
                    if "Waiting for" in text:
                        # For waiting messages, just show the full text
                        self.response_text.insert(tk.END, text, tag)
                    else:
                        # For answers or statements
                        if tag == "worker_message" and not text.startswith("Worker:"):
                            self.response_text.insert(tk.END, "Worker responds:\n", "worker_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        elif tag == "supervisor_message" and not text.startswith("Supervisor:"):
                            self.response_text.insert(tk.END, "Supervisor responds:\n", "supervisor_answer_header")
                            self.response_text.insert(tk.END, text, tag)
                        else:
                            self.response_text.insert(tk.END, text, tag)
                            
                # Make sure the new text is visible
                self.response_text.see(tk.END)
                
                # Reset to read-only
                self.response_text.config(state=tk.DISABLED)
                
                # Reset pending state
                self._pending_replace_id = None
                self._pending_message = None
                self._pending_tag = None
        except:
            # Reset pending state on any error
            self._pending_replace_id = None
            self._pending_message = None
            self._pending_tag = None