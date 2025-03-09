#!/usr/bin/env python3
"""Main application file for the VoiceCallApp system."""

import tkinter as tk

from voice_call_app import VoiceCallApp

def main():
    # Create the root window for the Worker app
    worker_root = tk.Tk()
    worker_root.title("Worker (Local)")
    worker_root.iconify()  # Start minimized
    
    # Create worker app instance
    worker_app = VoiceCallApp(worker_root, "llama3.2:1b", "Worker (Local)")
    worker_app.model_label = "Worker (Local)"
    
    # Position window
    worker_root.geometry("+100+100")
    
    # Deiconify after setup
    worker_root.deiconify()
    
    # Start the main loop
    worker_root.mainloop()

if __name__ == "__main__":
    main()
