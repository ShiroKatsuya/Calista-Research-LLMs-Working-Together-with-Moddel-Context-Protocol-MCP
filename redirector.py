import tkinter as tk
import io

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