import tkinter as tk
from tkinter import messagebox, PhotoImage

from datetime import datetime
from main import main

import threading
import queue

# Queue for communication between threads
input_queue = queue.Queue()
output_queue = queue.Queue()



class VoiceCallApp:
    # Class-level variables to track instances and call state
    instances = []
    active_call = None
    waiting_call = None

    def __init__(self, root, model_name, model_path):
        self.root = root
        self.model_name = model_name
        self.model_path = model_path
        self.root.title("AI Voice Call")
        self.root.geometry("400x650")
        self.root.configure(bg="#121B22")  
        
        self.call_active = False
        self.call_start_time = None
        self.call_duration = "00:00"
        self.connected_to = None
        
        # Add this instance to the class-level instances list
        VoiceCallApp.instances.append(self)
        
        self.setup_ui()
        
    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#121B22")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_top_bar()
        self.setup_profile_section()
        self.setup_status_section()
        self.setup_prompt_section()
        self.setup_control_buttons()
        
    def setup_top_bar(self):
        self.top_frame = tk.Frame(self.main_frame, bg="#1F2C34", height=60)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))
        self.secure_label = tk.Label(self.top_frame, text="End-to-end encrypted", font=("Segoe UI", 10), fg="#8696A0", bg="#1F2C34")
        self.secure_label.pack(pady=(10, 0))

    def setup_profile_section(self):
        self.profile_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.profile_frame.pack(pady=30)
        
        self.profile_circle = tk.Canvas(self.profile_frame, width=150, height=150, bg="#121B22", highlightthickness=0)
        self.profile_circle.pack()
        self.profile_circle.create_oval(10, 10, 140, 140, fill="#128C7E", outline="#25D366", width=2)
        self.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
        self.name_label = tk.Label(self.profile_frame, text="Voice Assistant", font=("Segoe UI", 24, "bold"), fg="white", bg="#121B22")
        self.name_label.pack(pady=(15, 5))
        
        self.model_label = tk.Label(self.profile_frame, text=self.model_name, font=("Segoe UI", 14), fg="#00BFA5", bg="#121B22")
        self.model_label.pack()

    def setup_status_section(self):
        self.status_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(self.status_frame, font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        self.status_label.pack()
        
        self.duration_label = tk.Label(self.status_frame, text="", font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        
        self.start_call_button = tk.Button(self.status_frame, text="Start Calling", font=("Segoe UI", 12), bg="#00BFA5", fg="white",
                                             command=self.toggle_call, padx=10, pady=5, relief=tk.RAISED, bd=0)
        self.start_call_button.pack(pady=10)
        
        self.stop_call_button = tk.Button(self.status_frame, text="Stop Calling", font=("Segoe UI", 12), bg="#FF0000", fg="white",
                                            command=self.end_call, padx=10, pady=5, relief=tk.RAISED, bd=0)

    def setup_prompt_section(self):
        self.prompt_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.prompt_label = tk.Label(self.prompt_frame, text="Your message:", font=("Segoe UI", 12), fg="#8696A0", bg="#121B22", anchor="w")



        self.prompt_label.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.text_entry = tk.Text(self.prompt_frame, font=("Segoe UI", 12), height=4, bg="#1F2C34", fg="white", insertbackground="white",
                                  bd=0, padx=10, pady=10)
        self.text_entry.pack(fill=tk.X, padx=10)

    def setup_control_buttons(self):
        self.controls_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        self.buttons_frame = tk.Frame(self.controls_frame, bg="#121B22")
        self.buttons_frame.pack(pady=20)
        
        button_size = 60
        self.mute_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Mute")
        self.mute_frame.pack(side=tk.LEFT, padx=15)
        
        self.call_button_frame = self.create_circular_button(self.buttons_frame, button_size + 20, "#00BFA5", "Call", command=self.toggle_call)
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
        text_input = self.text_entry.get("1.0", tk.END)
        if not text_input.strip():
            messagebox.showerror("Error", "Voice message is required!")
            return
            
        # Store the input text to avoid potential issues with variable scope in threads
        input_text = text_input.strip()
        
        # Run main() in a separate thread to avoid blocking the UI
        def run_main_in_thread():
            try:
                # Process the input in the background thread
                main(input_text)
                print(f"Processed: {input_text}")
                # Schedule UI updates to run on the main thread
                self.root.after(0, lambda: self.status_label.config(text="Processing complete"))
                self.root.after(0, lambda: print(f"Processed: {input_text}"))
            except Exception as e:
                # Handle errors and update UI from the main thread
                error_message = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {error_message}"))

                        
    
        
        # Update UI to show processing state
        self.status_label.config(text="Processing voice message...")
        
        # Create and start the thread
        import threading
        threading_thread = threading.Thread(target=run_main_in_thread)
        threading_thread.daemon = True  # Thread will exit when main program exits
        threading_thread.start()
        
        # Keep the UI responsive by processing events
        self.root.update_idletasks()
        # Check if there's already an active call
        if VoiceCallApp.active_call is not None:
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
                    break
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
        
        self.update_duration()
        
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#FF0000", "End", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)

    def end_call(self):
        if self.connected_to:
            other_app = self.connected_to
            # End call for both apps
            self._terminate_call()
            other_app._terminate_call()
            
            # Reset class-level call tracking
            VoiceCallApp.active_call = None
            self.connected_to = None
            other_app.connected_to = None
        elif VoiceCallApp.waiting_call == self:
            # Cancel waiting call
            VoiceCallApp.waiting_call = None
            self._terminate_call()

    def _terminate_call(self):
        text_input = self.text_entry.get("1.0", tk.END)
        messagebox.showinfo("Call Complete", f"Voice call ended!\nMessage: {text_input}")
        
        self.call_active = False
        self.call_start_time = None
        
        self.status_label.config(text="")
        self.duration_label.pack_forget()
        self.start_call_button.pack(pady=10)
        self.stop_call_button.pack_forget()
        
        self.text_entry.config(state=tk.NORMAL)
        self.text_entry.delete("1.0", tk.END)
        
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