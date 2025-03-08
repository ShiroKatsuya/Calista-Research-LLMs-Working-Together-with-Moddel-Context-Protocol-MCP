import tkinter as tk
from tkinter import messagebox, PhotoImage
import sys
import os
from datetime import datetime
import time



class VoiceCallApp2:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Voice Call")
        self.root.geometry("400x650")
        self.root.configure(bg="#121B22")  
        
        self.call_active = False
        self.call_start_time = None
        self.call_duration = "00:00"
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg="#121B22")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top bar with call info
        self.top_frame = tk.Frame(self.main_frame, bg="#1F2C34", height=60)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # "Secure call" label
        self.secure_label = tk.Label(self.top_frame, text="End-to-end encrypted", 
                                     font=("Segoe UI", 10), fg="#8696A0", bg="#1F2C34")
        self.secure_label.pack(pady=(10, 0))
        
        # Profile section
        self.profile_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.profile_frame.pack(pady=30)
        
        # Create a circle frame for profile picture
        self.profile_circle = tk.Canvas(self.profile_frame, width=150, height=150, 
                                       bg="#121B22", highlightthickness=0)
        self.profile_circle.pack()
        
        # Draw circle with gradient
        self.profile_circle.create_oval(10, 10, 140, 140, fill="#128C7E", outline="#25D366", width=2)
        
        # Add initials in the circle
        self.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
        # Name and status
        self.name_label = tk.Label(self.profile_frame, text="Voice Assistant", 
                                  font=("Segoe UI", 24, "bold"), fg="white", bg="#121B22")
        self.name_label.pack(pady=(15, 5))
        
        self.model_label = tk.Label(self.profile_frame, text="llama3.2:3b", 
                                   font=("Segoe UI", 14), fg="#00BFA5", bg="#121B22")
        self.model_label.pack()
        
        # Call status label
        self.status_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(self.status_frame,
                                    font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        self.status_label.pack()
        
        self.duration_label = tk.Label(self.status_frame, text="", 
                                      font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        
        # Start Call button
        self.start_call_button = tk.Button(self.status_frame, text="Start Calling", 
                                          font=("Segoe UI", 12), bg="#00BFA5", fg="white",
                                          command=self.toggle_call, padx=10, pady=5,
                                          relief=tk.RAISED, bd=0)
        self.start_call_button.pack(pady=10)
        
        # Stop Call button (hidden initially)
        self.stop_call_button = tk.Button(self.status_frame, text="Stop Calling", 
                                         font=("Segoe UI", 12), bg="#FF0000", fg="white",
                                         command=self.end_call, padx=10, pady=5,
                                         relief=tk.RAISED, bd=0)
        
        # Middle area for prompt
        self.prompt_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.prompt_label = tk.Label(self.prompt_frame, text="Your message:", 
                                    font=("Segoe UI", 12), fg="#8696A0", bg="#121B22", anchor="w")
        self.prompt_label.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.text_entry = tk.Text(self.prompt_frame, font=("Segoe UI", 12), height=4, 
                                 bg="#1F2C34", fg="white", insertbackground="white",
                                 bd=0, padx=10, pady=10)
        self.text_entry.pack(fill=tk.X, padx=10)
        
        # Bottom control buttons
        self.controls_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        # Button layout
        self.buttons_frame = tk.Frame(self.controls_frame, bg="#121B22")
        self.buttons_frame.pack(pady=20)
        
        # Create circular buttons
        button_size = 60
        
        # Mute button
        self.mute_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Mute")
        self.mute_frame.pack(side=tk.LEFT, padx=15)
        
        # Call/End button
        self.call_button_frame = self.create_circular_button(self.buttons_frame, button_size+20, "#00BFA5", "Call", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
        
        # Speaker button
        self.speaker_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Speaker")
        self.speaker_frame.pack(side=tk.LEFT, padx=15)
        
    def create_circular_button(self, parent, size, color, text, command=None):
        frame = tk.Frame(parent, bg="#121B22")
        
        # Create circular button
        button = tk.Canvas(frame, width=size, height=size, bg="#121B22", highlightthickness=0)
        button.pack()
        
        # Draw the circle
        button.create_oval(2, 2, size-2, size-2, fill=color, outline="", width=0)
        
        # Add an icon or text
        if text == "Call":
            button.create_text(size//2, size//2, text="📞", font=("Segoe UI", size//3), fill="white")
        elif text == "Mute":
            button.create_text(size//2, size//2, text="🎤", font=("Segoe UI", size//3), fill="white")
        elif text == "Speaker":
            button.create_text(size//2, size//2, text="🔊", font=("Segoe UI", size//3), fill="white")
        elif text == "End":
            button.create_text(size//2, size//2, text="📞", font=("Segoe UI", size//3), fill="white")
        
        # Make button clickable
        if command:
            button.bind("<Button-1>", lambda event: command())
        
        # Add text label
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
        
        # Validate input
        if not text_input.strip():
            messagebox.showerror("Error", "Voice message is required!")
            return
        
        # Start the call
        self.call_active = True
        self.call_start_time = datetime.now()
        
        # Update UI
        self.status_label.config(text="Connected To Model llama3.2:3b ")
        self.duration_label.pack()
        self.start_call_button.pack_forget()  # Hide the start call button
        self.stop_call_button.pack(pady=10)  # Show the stop call button
        
        # Disable the text entry during the call
        self.text_entry.config(state=tk.DISABLED)
        
        # Start duration timer
        self.update_duration()
        
        # Change call button to red
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#FF0000", "End", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
    
    def end_call(self):
        # Process the call
        text_input = self.text_entry.get("1.0", tk.END)
        messagebox.showinfo("Call Complete", f"Voice call processed successfully!\nMessage: {text_input}")
        
        # Reset UI
        self.call_active = False
        self.call_start_time = None
        
        self.status_label.config(text="Calling...")
        self.duration_label.pack_forget()
        self.start_call_button.pack(pady=10)  # Show the start call button again
        self.stop_call_button.pack_forget()  # Hide the stop call button
        
        # Re-enable the text entry
        self.text_entry.config(state=tk.NORMAL)
        self.text_entry.delete("1.0", tk.END)
        
        # Change button back to green
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#00BFA5", "Call", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
    
    def update_duration(self):
        if self.call_active and self.call_start_time:
            now = datetime.now()
            diff = now - self.call_start_time
            
            # Format as MM:SS
            minutes = diff.seconds // 60
            seconds = diff.seconds % 60
            self.call_duration = f"{minutes:02d}:{seconds:02d}"
            
            self.duration_label.config(text=self.call_duration)
            
            # Update every second
            self.root.after(1000, self.update_duration)


class VoiceCallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Voice Call")
        self.root.geometry("400x650")
        self.root.configure(bg="#121B22")  
        
        self.call_active = False
        self.call_start_time = None
        self.call_duration = "00:00"
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, bg="#121B22")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top bar with call info
        self.top_frame = tk.Frame(self.main_frame, bg="#1F2C34", height=60)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # "Secure call" label
        self.secure_label = tk.Label(self.top_frame, text="End-to-end encrypted", 
                                     font=("Segoe UI", 10), fg="#8696A0", bg="#1F2C34")
        self.secure_label.pack(pady=(10, 0))
        
        # Profile section
        self.profile_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.profile_frame.pack(pady=30)
        
        # Create a circle frame for profile picture
        self.profile_circle = tk.Canvas(self.profile_frame, width=150, height=150, 
                                       bg="#121B22", highlightthickness=0)
        self.profile_circle.pack()
        
        # Draw circle with gradient
        self.profile_circle.create_oval(10, 10, 140, 140, fill="#128C7E", outline="#25D366", width=2)
        
        # Add initials in the circle
        self.profile_circle.create_text(75, 75, text="AI", font=("Segoe UI", 50, "bold"), fill="white")
        
        # Name and status
        self.name_label = tk.Label(self.profile_frame, text="Voice Assistant", 
                                  font=("Segoe UI", 24, "bold"), fg="white", bg="#121B22")
        self.name_label.pack(pady=(15, 5))
        
        self.model_label = tk.Label(self.profile_frame, text="deepseek-r1:1.5b", 
                                   font=("Segoe UI", 14), fg="#00BFA5", bg="#121B22")
        self.model_label.pack()
        
        # Call status label
        self.status_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.status_frame.pack(pady=10)
        
        self.status_label = tk.Label(self.status_frame,
                                    font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        self.status_label.pack()
        
        self.duration_label = tk.Label(self.status_frame, text="", 
                                      font=("Segoe UI", 14), fg="#8696A0", bg="#121B22")
        
        # Start Call button
        self.start_call_button = tk.Button(self.status_frame, text="Start Calling", 
                                          font=("Segoe UI", 12), bg="#00BFA5", fg="white",
                                          command=self.toggle_call, padx=10, pady=5,
                                          relief=tk.RAISED, bd=0)
        self.start_call_button.pack(pady=10)
        
        # Stop Call button (hidden initially)
        self.stop_call_button = tk.Button(self.status_frame, text="Stop Calling", 
                                         font=("Segoe UI", 12), bg="#FF0000", fg="white",
                                         command=self.end_call, padx=10, pady=5,
                                         relief=tk.RAISED, bd=0)
        
        # Middle area for prompt
        self.prompt_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.prompt_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.prompt_label = tk.Label(self.prompt_frame, text="Your message:", 
                                    font=("Segoe UI", 12), fg="#8696A0", bg="#121B22", anchor="w")
        self.prompt_label.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.text_entry = tk.Text(self.prompt_frame, font=("Segoe UI", 12), height=4, 
                                 bg="#1F2C34", fg="white", insertbackground="white",
                                 bd=0, padx=10, pady=10)
        self.text_entry.pack(fill=tk.X, padx=10)
        
        # Bottom control buttons
        self.controls_frame = tk.Frame(self.main_frame, bg="#121B22")
        self.controls_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        # Button layout
        self.buttons_frame = tk.Frame(self.controls_frame, bg="#121B22")
        self.buttons_frame.pack(pady=20)
        
        # Create circular buttons
        button_size = 60
        
        # Mute button
        self.mute_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Mute")
        self.mute_frame.pack(side=tk.LEFT, padx=15)
        
        # Call/End button
        self.call_button_frame = self.create_circular_button(self.buttons_frame, button_size+20, "#00BFA5", "Call", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
        
        # Speaker button
        self.speaker_frame = self.create_circular_button(self.buttons_frame, button_size, "#262D31", "Speaker")
        self.speaker_frame.pack(side=tk.LEFT, padx=15)
        
    def create_circular_button(self, parent, size, color, text, command=None):
        frame = tk.Frame(parent, bg="#121B22")
        
        # Create circular button
        button = tk.Canvas(frame, width=size, height=size, bg="#121B22", highlightthickness=0)
        button.pack()
        
        # Draw the circle
        button.create_oval(2, 2, size-2, size-2, fill=color, outline="", width=0)
        
        # Add an icon or text
        if text == "Call":
            button.create_text(size//2, size//2, text="📞", font=("Segoe UI", size//3), fill="white")
        elif text == "Mute":
            button.create_text(size//2, size//2, text="🎤", font=("Segoe UI", size//3), fill="white")
        elif text == "Speaker":
            button.create_text(size//2, size//2, text="🔊", font=("Segoe UI", size//3), fill="white")
        elif text == "End":
            button.create_text(size//2, size//2, text="📞", font=("Segoe UI", size//3), fill="white")
        
        # Make button clickable
        if command:
            button.bind("<Button-1>", lambda event: command())
        
        # Add text label
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
        
        # Validate input
        if not text_input.strip():
            messagebox.showerror("Error", "Voice message is required!")
            return
        
        # Start the call
        self.call_active = True
        self.call_start_time = datetime.now()
        
        # Update UI
        self.status_label.config(text="Connected To Model llama3.2:3b ")
        self.duration_label.pack()
        self.start_call_button.pack_forget()  # Hide the start call button
        self.stop_call_button.pack(pady=10)  # Show the stop call button
        
        # Disable the text entry during the call
        self.text_entry.config(state=tk.DISABLED)
        
        # Start duration timer
        self.update_duration()
        
        # Change call button to red
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#FF0000", "End", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
    
    def end_call(self):
        # Process the call
        text_input = self.text_entry.get("1.0", tk.END)
        messagebox.showinfo("Call Complete", f"Voice call processed successfully!\nMessage: {text_input}")
        
        # Reset UI
        self.call_active = False
        self.call_start_time = None
        
        self.status_label.config(text="Calling...")
        self.duration_label.pack_forget()
        self.start_call_button.pack(pady=10)  # Show the start call button again
        self.stop_call_button.pack_forget()  # Hide the stop call button
        
        # Re-enable the text entry
        self.text_entry.config(state=tk.NORMAL)
        self.text_entry.delete("1.0", tk.END)
        
        # Change button back to green
        self.call_button_frame.destroy()
        self.call_button_frame = self.create_circular_button(self.buttons_frame, 80, "#00BFA5", "Call", command=self.toggle_call)
        self.call_button_frame.pack(side=tk.LEFT, padx=15)
    
    def update_duration(self):
        if self.call_active and self.call_start_time:
            now = datetime.now()
            diff = now - self.call_start_time
            
            # Format as MM:SS
            minutes = diff.seconds // 60
            seconds = diff.seconds % 60
            self.call_duration = f"{minutes:02d}:{seconds:02d}"
            
            self.duration_label.config(text=self.call_duration)
            
            # Update every second
            self.root.after(1000, self.update_duration)

if __name__ == "__main__":
    app_root1 = tk.Tk()
    app1 = VoiceCallApp(app_root1)
    
    app_root2 = tk.Tk()
    app2 = VoiceCallApp2(app_root2)
    
    # Run both windows simultaneously
    app_root1.update()
    app_root2.update()
    
    while True:
        try:
            app_root1.update()
            app_root2.update()
        except tk.TkinterError:
            break

    
