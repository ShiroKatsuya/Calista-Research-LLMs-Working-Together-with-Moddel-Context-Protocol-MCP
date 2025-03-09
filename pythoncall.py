import tkinter as tk
from template import VoiceCallApp

if __name__ == "__main__":
    main_root = tk.Tk()
    main_root.title("Supervisor (Remote)")
    app1 = VoiceCallApp(main_root, model_name="llama3.2:3b", model_path="deepseek-r1:1.5b")
    app1.model_label = "Supervisor (Remote)"  # Label for the first app
    app1.name_label.config(text=app1.model_label)  # Update the label display
    
    second_window = tk.Toplevel(main_root)
    second_window.title("Worker (Local)")
    app2 = VoiceCallApp(second_window, model_name="deepseek-r1:1.5b", model_path="llama3.2:3b")
    app2.model_label = "Worker (Local)"  # Label for the second app
    app2.name_label.config(text=app2.model_label)  # Update the label display
    
    # Explicitly establish connection between the two apps
    app1.connected_to = app2
    app2.connected_to = app1
    
    # Set as active call to enable proper communication
    VoiceCallApp.active_call = app1
    
    main_root.mainloop()

    
