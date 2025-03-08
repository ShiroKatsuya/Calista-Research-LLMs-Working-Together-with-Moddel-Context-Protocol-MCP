import tkinter as tk
from tkinter import messagebox
import sys
import os

def process_voice_call():
    text_input = text_entry.get("1.0", tk.END)
    
    # Validate input
    if not text_input.strip():
        messagebox.showerror("Error", "Voice prompt is required!")
        return
    
    messagebox.showinfo("Call Complete", f"Voice call processed successfully!\nPrompt: {text_input}")
    
    text_entry.delete("1.0", tk.END)


root = tk.Tk()
root.title("AI Voice Call")
root.geometry("400x650")  # More phone-like dimensions
root.configure(bg="#075E54")  # WhatsApp dark green color


# Main frame with WhatsApp style
call_frame = tk.Frame(root, bg="#DCF8C6")  # WhatsApp light green chat background
call_frame.pack(fill=tk.BOTH, expand=True)

# Top bar with WhatsApp style
top_bar = tk.Frame(call_frame, bg="#075E54", height=60)  # WhatsApp dark green header
top_bar.pack(fill=tk.X)
top_bar.pack_propagate(False)

# Back arrow and profile picture placeholder
back_btn = tk.Label(top_bar, text="←", font=("Arial", 16), bg="#075E54", fg="white")
back_btn.pack(side=tk.LEFT, padx=10)

# Profile picture (circle placeholder)
profile_frame = tk.Frame(top_bar, bg="#075E54", width=40, height=40)
profile_frame.pack(side=tk.LEFT, padx=5)
profile_pic = tk.Canvas(profile_frame, width=40, height=40, bg="#075E54", highlightthickness=0)
profile_pic.pack()
profile_pic.create_oval(5, 5, 35, 35, fill="#128C7E", outline="")

# Contact name
header = tk.Label(top_bar, font=("Arial", 14, "bold"), text="AI Assistant", 
                 bg="#075E54", fg="white")
header.pack(side=tk.LEFT, padx=10)

# Call icons on right
call_icons = tk.Frame(top_bar, bg="#075E54")
call_icons.pack(side=tk.RIGHT)
video_icon = tk.Label(call_icons, text="📹", font=("Arial", 16), bg="#075E54", fg="white")
video_icon.pack(side=tk.LEFT, padx=5)
more_icon = tk.Label(call_icons, text="⋮", font=("Arial", 16), bg="#075E54", fg="white")
more_icon.pack(side=tk.LEFT, padx=10)

# Model info
model_label = tk.Label(call_frame, font=("Arial", 10), text="deepseek-r1:1.5b", 
                      bg="#DCF8C6", fg="#128C7E")
model_label.pack(anchor="center", pady=5)

# Chat area
chat_area = tk.Frame(call_frame, bg="#DCF8C6")
chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# System message bubble
system_msg_frame = tk.Frame(chat_area, bg="#E1FFC7", padx=10, pady=8, bd=1, relief=tk.SOLID)
system_msg_frame.pack(anchor="e", pady=5, padx=5)
system_msg = tk.Label(system_msg_frame, text="How can I help you today?", 
                     font=("Arial", 11), bg="#E1FFC7", fg="#000000", wraplength=250)
system_msg.pack()

# Text entry area at bottom
input_area = tk.Frame(call_frame, bg="#075E54", height=60)
input_area.pack(fill=tk.X, side=tk.BOTTOM)
input_area.pack_propagate(False)

text_entry = tk.Text(input_area, font=("Arial", 12), width=25, height=2, 
                    bg="white", fg="black", insertbackground="black", bd=0,
                    padx=10, pady=5)
text_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

# Create a loading indicator that will be shown during call processing
loading_indicator = tk.Label(input_area, text="Recording...", 
                           font=("Arial", 10), bg="#075E54", fg="white")
loading_indicator.pack(side=tk.RIGHT, padx=5)
loading_indicator.pack_forget()  # Hide initially

def call_with_animation():
    # Show loading indicator
    call_button.pack_forget()
    loading_indicator.pack(side=tk.RIGHT, padx=5)
    root.update()
    
    # Simulate processing delay
    root.after(1500, lambda: [process_voice_call(), 
                             loading_indicator.pack_forget(),
                             call_button.pack(side=tk.RIGHT, padx=10, pady=10)])

# Voice call button (microphone icon)
call_button = tk.Button(input_area, text="🎤", font=("Arial", 16), 
                       bg="#128C7E", fg="white", width=3, height=1,
                       command=call_with_animation, bd=0)
call_button.pack(side=tk.RIGHT, padx=10, pady=10)


if __name__ == "__main__":
    root.mainloop()
