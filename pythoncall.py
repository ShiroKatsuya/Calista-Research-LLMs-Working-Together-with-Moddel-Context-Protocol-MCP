import tkinter as tk
from template import VoiceCallApp


if __name__ == "__main__":
    main_root = tk.Tk()
    app1 = VoiceCallApp(main_root, model_name=" llama3.2:3b", model_path="deepseek-r1:1.5b")
    
    second_window = tk.Toplevel(main_root)
    app2 = VoiceCallApp(second_window, model_name="deepseek-r1:1.5b", model_path="llama3.2:3b")
    
    main_root.mainloop()

    
