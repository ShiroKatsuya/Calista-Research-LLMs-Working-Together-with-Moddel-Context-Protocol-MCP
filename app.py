#!/usr/bin/env python3
"""Main application file for the VoiceCallApp system."""

import tkinter as tk

from voice_call_app import VoiceCallApp

def main():
    # Create the root window for the first app (Worker)
    worker_root = tk.Tk()
    worker_root.title("Worker (Local)")
    worker_root.iconify()  # Start minimized

    # Create the root window for the second app (Supervisor)
    supervisor_root = tk.Toplevel()
    supervisor_root.title("Supervisor (Remote)")
    
    # Create app instances
    worker_app = VoiceCallApp(worker_root, "llama3.2:1b", "Worker (Local)")
    worker_app.model_label = "Worker (Local)"
    
    supervisor_app = VoiceCallApp(supervisor_root, "llama3.2:3b", "Supervisor (Remote)")
    supervisor_app.model_label = "Supervisor (Remote)"
    
    # Position windows
    worker_root.geometry("+100+100")
    supervisor_root.geometry("+550+100")
    
    # Deiconify after setup
    worker_root.deiconify()
    
    # Start the main loop
    worker_root.mainloop()

if __name__ == "__main__":
    main()
