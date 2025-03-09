import tkinter as tk
from tkinter import messagebox
import sys
import threading
import traceback
from datetime import datetime
from contextlib import redirect_stdout
# Don't import VoiceCallApp directly to avoid circular import
from message_handlers import MessageHandlers
from redirector import StdoutRedirector

class CallHandlers:
    # Class-level list to track app instances (moved from VoiceCallApp)
    instances = []
    
    # Track conversation content for terminal display
    worker_conversation = []
    supervisor_conversation = []
    
    @staticmethod
    def print_to_terminal(message, sender=None, is_system=False):
        """Print a message to the terminal without any truncation."""
        # Store in conversation history
        if sender == "Worker":
            CallHandlers.worker_conversation.append(message)
        elif sender == "Supervisor":
            CallHandlers.supervisor_conversation.append(message)
        
        # Add appropriate prefix/formatting
        if is_system:
            print(f"\n[SYSTEM] {message}\n")
        elif sender == "Worker":
            print(f"\n👨‍💻 Worker says:\n{message}\n")
        elif sender == "Supervisor":
            print(f"\n👩‍💼 Supervisor says:\n{message}\n")
        else:
            print(f"\n{message}\n")
    
    @staticmethod
    def print_full_conversation():
        """Print the entire conversation history to the terminal."""
        print("\n" + "=" * 80)
        print("FULL CONVERSATION HISTORY")
        print("=" * 80)
        
        # Print messages in chronological order with proper formatting
        for i in range(max(len(CallHandlers.worker_conversation), len(CallHandlers.supervisor_conversation))):
            # Print worker message if available
            if i < len(CallHandlers.worker_conversation):
                print(f"\n👨‍💻 Worker says:\n{CallHandlers.worker_conversation[i]}\n")
                print("-" * 40)
            
            # Print supervisor message if available
            if i < len(CallHandlers.supervisor_conversation):
                print(f"\n👩‍💼 Supervisor says:\n{CallHandlers.supervisor_conversation[i]}\n")
                print("-" * 40)
        
        print("=" * 80 + "\n")
    
    @staticmethod
    def toggle_call(app, event=None):
        """Toggle call state."""
        if app.call_active:
            CallHandlers.end_call(app)
        else:
            CallHandlers.start_call(app)

    @staticmethod
    def start_call(app):
        """Start a call with another instance."""
        # Don't start if already on a call
        if app.call_active:
            return
        
        # Clear conversation tracking at the start of a new call
        CallHandlers.worker_conversation = []
        CallHandlers.supervisor_conversation = []
        
        # Print start of conversation to terminal
        CallHandlers.print_to_terminal("Starting new conversation", is_system=True)
        
        # Get text from input field
        if hasattr(app, 'text_entry'):
            input_text = app.text_entry.get("1.0", "end-1c").strip()
            # Skip placeholder text
            if input_text == "Type a message...":
                input_text = ""
                
            # Check if there's actually text to process
            if not input_text:
                messagebox.showwarning("Input Required", "Please type a message before starting the call.")
                return
                
            # Clear the input field after getting text
            app.text_entry.delete("1.0", tk.END)
            
            # Process the input in a separate thread to avoid blocking the UI
            def process_input():
                try:
                    # Create a custom redirector to capture output
                    class ConversationOutputRedirector:
                        def __init__(self, app_instance):
                            self.app_instance = app_instance
                            self.buffer = ""
                            self.full_output = []  # Store all output for complete display
                            
                        def write(self, string):
                            # Always save the complete raw output
                            self.full_output.append(string)
                            
                            # Also process line by line for UI updates
                            self.buffer += string
                            if '\n' in string:
                                lines = self.buffer.split('\n')
                                # Process all complete lines except the last (which might be incomplete)
                                for line in lines[:-1]:
                                    self._process_line(line)
                                # Keep the last (potentially incomplete) line in the buffer
                                self.buffer = lines[-1]
                            
                        def _process_line(self, line):
                            if not line.strip():
                                return  # Skip empty lines
                                
                            # Print the raw line to terminal without processing
                            # This ensures nothing is cut off
                            print(line)
                                
                            # Process for UI updates (run in non-blocking way)
                            app = self.app_instance
                            
                            # Use try/except to prevent freezing if UI update fails
                            try:
                                # Process based on line content
                                if "Supervisor (Remote) is thinking" in line:
                                    # Show supervisor thinking state in both apps
                                    app.root.after(0, lambda: MessageHandlers.show_thinking_in_response(app, True))
                                    if app.connected_to:
                                        app.connected_to.root.after(0, lambda: MessageHandlers.show_thinking_in_response(app.connected_to, True))
                                
                                elif "Worker (Local) is thinking" in line:
                                    # Show worker thinking state in both apps
                                    app.root.after(0, lambda: MessageHandlers.show_thinking_in_response(app, True))
                                    if app.connected_to:
                                        app.connected_to.root.after(0, lambda: MessageHandlers.show_thinking_in_response(app.connected_to, True))
                                
                                elif line.startswith("@Worker:"):
                                    # Worker message - parse and store
                                    message = line.replace("@Worker:", "").strip()
                                    CallHandlers.worker_conversation.append(message)
                                    
                                    # Update UI safely in a non-blocking way
                                    self._safe_ui_update_for_worker(message)
                                
                                elif line.startswith("@Supervisor:") or "Supervisor (Remote) answers:" in line:
                                    # Supervisor message - parse and store
                                    message = line.replace("@Supervisor:", "").replace("Supervisor (Remote) answers:", "").strip()
                                    CallHandlers.supervisor_conversation.append(message)
                                    
                                    # Update UI safely in a non-blocking way
                                    self._safe_ui_update_for_supervisor(message)
                                
                                elif "<|start_header_id|>" in line or "<|end_header_id|>" in line:
                                    # Filter out header markers from the display
                                    pass
                                
                                elif "⚡ I think" in line or "⚡ In my opinion" in line:
                                    # Worker insights - highlight and store
                                    CallHandlers.worker_conversation.append(line)
                                    self._safe_ui_update_for_worker(line)
                                
                                else:
                                    # General content - try to categorize
                                    if "Worker (Local)" in line:
                                        self._safe_ui_update_for_worker(line)
                                    elif "Supervisor (Remote)" in line:
                                        self._safe_ui_update_for_supervisor(line)
                                    else:
                                        # System message - display in both apps if not empty
                                        content = line.strip()
                                        if content:
                                            self._safe_ui_update_for_system(content)
                            except Exception as e:
                                # Prevent UI freezing if an error occurs during processing
                                print(f"Error processing line: {e}")
                        
                        def _safe_ui_update_for_worker(self, message):
                            """Updates the worker app UI without blocking the main thread."""
                            app = self.app_instance
                            
                            # Find the right apps
                            worker_app = None
                            supervisor_app = None
                            
                            for instance in CallHandlers.instances:
                                if instance.model_label == "Worker (Local)":
                                    worker_app = instance
                                elif instance.model_label == "Supervisor (Remote)":
                                    supervisor_app = instance
                            
                            # Schedule UI updates with delay to prevent flooding
                            if worker_app:
                                worker_app.root.after(10, lambda msg=message: 
                                                   self._update_if_exists(worker_app, msg))
                            
                            if supervisor_app:
                                supervisor_app.root.after(10, lambda msg=message: 
                                                      self._update_if_exists(supervisor_app, f"Worker: {msg}"))
                        
                        def _safe_ui_update_for_supervisor(self, message):
                            """Updates the supervisor app UI without blocking the main thread."""
                            app = self.app_instance
                            
                            # Find the right apps
                            worker_app = None
                            supervisor_app = None
                            
                            for instance in CallHandlers.instances:
                                if instance.model_label == "Worker (Local)":
                                    worker_app = instance
                                elif instance.model_label == "Supervisor (Remote)":
                                    supervisor_app = instance
                            
                            # Schedule UI updates with delay to prevent flooding  
                            if supervisor_app:
                                supervisor_app.root.after(10, lambda msg=message: 
                                                      self._update_if_exists(supervisor_app, msg))
                            
                            if worker_app:
                                worker_app.root.after(10, lambda msg=message: 
                                                   self._update_if_exists(worker_app, f"Supervisor: {msg}"))
                        
                        def _safe_ui_update_for_system(self, message):
                            """Updates all apps with system messages without blocking."""
                            for instance in CallHandlers.instances:
                                if instance.root.winfo_exists():
                                    instance.root.after(10, lambda i=instance, msg=message: 
                                                     self._update_if_exists(i, msg))
                        
                        def _update_if_exists(self, app, message):
                            """Safe wrapper to update UI only if components still exist."""
                            try:
                                if app.root.winfo_exists() and hasattr(app, 'response_text') and app.response_text.winfo_exists():
                                    MessageHandlers.append_to_response(app, message)
                            except Exception as e:
                                print(f"Error updating UI: {e}")
                        
                        def flush(self):
                            # If there's anything left in the buffer, process it
                            if self.buffer:
                                self._process_line(self.buffer)
                                self.buffer = ""
                            
                            # Print the complete raw output to terminal
                            print("\n" + "=" * 80)
                            print("COMPLETE RAW OUTPUT")
                            print("=" * 80)
                            for chunk in self.full_output:
                                sys.stdout.write(chunk)
                            print("=" * 80)
                    
                    # Use a try-finally structure to properly clean up resources
                    import sys
                    original_stdout = sys.stdout
                    redirector = ConversationOutputRedirector(app)
                    sys.stdout = redirector
                    
                    try:
                        # Import main function here to avoid circular imports
                        from main import main
                        # Process the input text - add timeout mechanism
                        main_thread = threading.Thread(target=lambda: main(input_text))
                        main_thread.daemon = True  # Allow thread to be terminated when app closes
                        main_thread.start()
                        
                        # Add a watchdog to check if processing is taking too long
                        def check_processing_status():
                            if main_thread.is_alive():
                                # Still processing, update UI to show it's still working
                                if app.root.winfo_exists():
                                    app.status_label.config(text="Still processing...")
                                    # Schedule another check
                                    app.root.after(1000, check_processing_status)
                            else:
                                # Processing complete
                                if app.root.winfo_exists():
                                    app.status_label.config(text="Processing complete")
                        
                        # Start the watchdog after a short delay
                        app.root.after(3000, check_processing_status)
                        
                    except Exception as e:
                        print(f"Error processing input: {e}")
                        traceback.print_exc()  # Print stack trace for debugging
                        if hasattr(app, 'status_label') and app.root.winfo_exists():
                            app.status_label.config(text=f"Error: {str(e)}")
                    finally:
                        # Always restore stdout, but do it in the main thread
                        app.root.after(0, lambda: setattr(sys, 'stdout', original_stdout))
                        # Flush any remaining content
                        app.root.after(0, redirector.flush)
                except Exception as e:
                    # Outer exception handler for the entire process_input function
                    print(f"Fatal error in process_input: {e}")
                    traceback.print_exc()
                    if hasattr(app, 'status_label') and app.root.winfo_exists():
                        app.status_label.config(text=f"Fatal error: {str(e)}")
            
            # Start a thread to process the input
            input_thread = threading.Thread(target=process_input)
            input_thread.daemon = True
            input_thread.start()
        
        # Update UI to show connecting state
        app.status_label.config(text="Connecting...")
        
        # Update status indicator to yellow (connecting)
        if hasattr(app, 'status_indicator'):
            app.status_indicator.itemconfig("indicator", fill="#FFC107")
            
            # Create pulsing effect for connecting indicator
            def pulse_indicator():
                if not app.call_active or not app.status_indicator.winfo_exists():
                    return
                current_fill = app.status_indicator.itemcget("indicator", "fill")
                new_fill = "#FFA000" if current_fill == "#FFC107" else "#FFC107"
                app.status_indicator.itemconfig("indicator", fill=new_fill)
                app.root.after(500, pulse_indicator)
                
            app.root.after(100, pulse_indicator)
        
        # Hide start button, show stop button
        app.start_call_button.pack_forget()
        app.stop_call_button.pack(pady=10)
        
        # Show the duration label
        app.duration_label.pack()
        app.call_duration = 0
        app.duration_label.config(text="00:00")
        
        # Set the call active flag
        app.call_active = True
        app.is_thinking = True
        
        # Show thinking state in this app
        MessageHandlers.show_thinking_in_response(app, True)
        
        if app.connected_to:
            MessageHandlers.show_thinking_in_response(app.connected_to, True)
            
        # Get current timestamp
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Add a call start message with proper icon and timestamp
        app.response_text.config(state=tk.NORMAL)
        if MessageHandlers.preserve_history and app.response_text.get("1.0", tk.END).strip():
            app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
        app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.CALL_EMOJI} Call starting...\n", "system_header")
        app.response_text.config(state=tk.DISABLED)
        
        # Make sure the call message is visible
        MessageHandlers._ensure_autoscroll(app)
        
        # Update UI in button to show active call
        if hasattr(app, 'call_button_frame'):
            for widget in app.call_button_frame.winfo_children():
                if isinstance(widget, tk.Canvas):
                    widget.itemconfig("button_bg", fill="#FF0000")  # Change to red for active call
                    # Remove phone icon, add end call icon
                    widget.delete("button_icon")
                    widget.create_text(widget.winfo_width() // 2, 
                                     widget.winfo_height() // 2, 
                                     text="❌", 
                                     font=("Segoe UI Emoji", widget.winfo_width() // 3), 
                                     fill="white", 
                                     tags="button_icon")
        
        # Start the timer to track call duration
        app.root.after(1000, lambda: CallHandlers.update_duration(app))
        
        # Find another instance to establish a connection and auto-accept
        receiver_instance = None
        for instance in CallHandlers.instances:
            if instance != app and not instance.call_active:
                receiver_instance = instance
                # Connect the two apps
                app.connected_to = instance
                instance.connected_to = app
                
                # Auto-accept the call in the receiving app
                CallHandlers._auto_accept_call(instance, app)
                break
                
        # If we found a receiver, establish the connection
        if receiver_instance:
            app.root.after(500, lambda: CallHandlers._establish_call(app, receiver_instance))

    @staticmethod
    def _auto_accept_call(receiving_app, calling_app):
        """Automatically accept an incoming call in the receiving app."""
        # Update status to show that call is connected
        receiving_app.status_label.config(text="Call connected")
        
        # Update status indicator to green (active)
        if hasattr(receiving_app, 'status_indicator'):
            receiving_app.status_indicator.itemconfig("indicator", fill="#4CAF50")
            
        # Update UI to show call is active
        receiving_app.call_active = True
        receiving_app.is_thinking = True
            
        # Hide start button, show stop button
        receiving_app.start_call_button.pack_forget()
        receiving_app.stop_call_button.pack(pady=10)
        
        # Show the duration label and initialize call duration
        receiving_app.duration_label.pack()
        receiving_app.call_duration = 0
        receiving_app.duration_label.config(text="00:00")
        
        # Update UI in button to show active call
        if hasattr(receiving_app, 'call_button_frame'):
            for widget in receiving_app.call_button_frame.winfo_children():
                if isinstance(widget, tk.Canvas):
                    widget.itemconfig("button_bg", fill="#FF0000")  # Change to red for active call
                    # Remove phone icon, add end call icon
                    widget.delete("button_icon")
                    widget.create_text(widget.winfo_width() // 2, 
                                     widget.winfo_height() // 2, 
                                     text="❌", 
                                     font=("Segoe UI Emoji", widget.winfo_width() // 3), 
                                     fill="white", 
                                     tags="button_icon")
                    
        # Get current timestamp
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Add a call start message to the receiving app
        receiving_app.response_text.config(state=tk.NORMAL)
        if MessageHandlers.preserve_history and receiving_app.response_text.get("1.0", tk.END).strip():
            receiving_app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
        receiving_app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.CALL_EMOJI} Call automatically connected with {calling_app.model_label}\n", "system_header")
        receiving_app.response_text.config(state=tk.DISABLED)
        
        # Make sure the call message is visible
        MessageHandlers._ensure_autoscroll(receiving_app)
        
        # Start the timer to track call duration in the receiving app
        receiving_app.root.after(1000, lambda: CallHandlers.update_duration(receiving_app))

    @staticmethod
    def _establish_call(app, other_app):
        """Set up the UI for an established call."""
        from message_handlers import MessageHandlers
        
        app.call_active = True
        import datetime
        app.call_start_time = datetime.datetime.now()
        app.status_label.config(text=f"Connected to {other_app.model_name}")
        app.duration_label.pack()
        app.start_call_button.pack_forget()
        app.stop_call_button.pack(pady=10)
        app.text_entry.config(state=tk.DISABLED)
        
        # Add a connected message with timestamp
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Clear any previous response first but respect history
        app.response_text.config(state=tk.NORMAL)
        
        # Remove existing thinking indicators
        MessageHandlers._remove_thinking_indicators(app)
        
        # Add connection message with proper formatting 
        if MessageHandlers.preserve_history and app.response_text.get("1.0", tk.END).strip():
            app.response_text.insert(tk.END, "\n\n", "")
        
        app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.CALL_EMOJI} Connected to {other_app.model_name}\n", "system_header")
        app.response_text.config(state=tk.DISABLED)
        
        # Show appropriate thinking status
        MessageHandlers.show_thinking_in_response(app, True)
        
        # Start the duration timer
        CallHandlers.update_duration(app)
        
        # Change call button to red for ending the call
        app.call_button_frame.destroy()
        from ui_components import UIComponents
        app.call_button_frame = UIComponents.create_circular_button(
            app,
            app.buttons_frame, 
            80, 
            "#FF0000", 
            "End", 
            command=app.toggle_call
        )
        app.call_button_frame.pack(side=tk.LEFT, padx=15)

    @staticmethod
    def end_call(app):
        """End the active call."""
        if not app.call_active:
            return
            
        # Set the ending call flag to prevent multiple end_call calls
        if app.is_ending_call:
            return
        app.is_ending_call = True
        
        # Get the connected app (if any)
        connected_app = app.connected_to
        
        # Update status
        app.status_label.config(text="Ending call...")
        if hasattr(app, 'status_indicator'):
            app.status_indicator.itemconfig("indicator", fill="#FF5722")  # Orange for ending call
        
        if connected_app and connected_app.root.winfo_exists():
            connected_app.status_label.config(text="Call ending...")
            if hasattr(connected_app, 'status_indicator'):
                connected_app.status_indicator.itemconfig("indicator", fill="#FF5722")  # Orange for ending call
        
        # Store current conversation state
        worker_response = None
        supervisor_response = None
        if app.response_text.winfo_exists():
            current_text = app.response_text.get("1.0", tk.END)
            
            # Save conversation history before ending the call
            MessageHandlers.save_conversation_history(app)
            
            # Store for connected app
            if app.model_label == "Worker (Local)":
                worker_response = current_text
            elif app.model_label == "Supervisor (Remote)":
                supervisor_response = current_text
                
        if connected_app and connected_app.response_text.winfo_exists():
            connected_text = connected_app.response_text.get("1.0", tk.END)
            
            # Save conversation history for connected app
            MessageHandlers.save_conversation_history(connected_app)
            
            # Store for later
            if connected_app.model_label == "Worker (Local)":
                worker_response = connected_text
            elif connected_app.model_label == "Supervisor (Remote)":
                supervisor_response = connected_text
    
        # Print the complete conversation to the terminal
        CallHandlers.print_to_terminal("Call ended - Full conversation:", is_system=True)
        CallHandlers.print_full_conversation()
        
        # Save the conversation to a text file
        text_file = CallHandlers.save_conversation_to_text_file()
        if text_file:
            app.status_label.config(text=f"Conversation saved to {text_file}")
            if connected_app and connected_app.root.winfo_exists():
                connected_app.status_label.config(text=f"Conversation saved to {text_file}")
        
        # Add a call end message
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Add call ended message with timestamp
        app.response_text.config(state=tk.NORMAL)
        app.response_text.insert(tk.END, f"\n[{current_time}] {MessageHandlers.CALL_EMOJI} Call ended ", "system_header")
        
        # Add call duration
        if hasattr(app, 'call_duration'):
            minutes = app.call_duration // 60
            seconds = app.call_duration % 60
            app.response_text.insert(tk.END, f"• Duration: {minutes:02d}:{seconds:02d}\n", "call_info")
        else:
            app.response_text.insert(tk.END, "\n", "")
            
        app.response_text.config(state=tk.DISABLED)
        
        # Do the same for connected app
        if connected_app and connected_app.response_text.winfo_exists():
            connected_app.response_text.config(state=tk.NORMAL)
            connected_app.response_text.insert(tk.END, f"\n[{current_time}] {MessageHandlers.CALL_EMOJI} Call ended ", "system_header")
            
            # Add call duration
            if hasattr(connected_app, 'call_duration'):
                minutes = connected_app.call_duration // 60
                seconds = connected_app.call_duration % 60
                connected_app.response_text.insert(tk.END, f"• Duration: {minutes:02d}:{seconds:02d}\n", "call_info")
            else:
                connected_app.response_text.insert(tk.END, "\n", "")
                
            connected_app.response_text.config(state=tk.DISABLED)
            
        # Proceed with call termination
        CallHandlers._complete_end_call(app)

    @staticmethod
    def _complete_end_call(app):
        """Complete the call ending process."""
        app.call_active = False
        
        # Update UI
        app.stop_call_button.pack_forget()
        app.start_call_button.pack(pady=10)
        
        # Update status indicator to idle (green)
        if hasattr(app, 'status_indicator'):
            app.status_indicator.itemconfig("indicator", fill="#4CAF50")
            
        # Restore call button to normal state
        if hasattr(app, 'call_button_frame'):
            for widget in app.call_button_frame.winfo_children():
                if isinstance(widget, tk.Canvas):
                    widget.itemconfig("button_bg", fill="#00BFA5")  # Reset to green
                    # Remove end call icon, add phone icon
                    widget.delete("button_icon")
                    widget.create_text(widget.winfo_width() // 2, 
                                     widget.winfo_height() // 2, 
                                     text="📞", 
                                     font=("Segoe UI Emoji", widget.winfo_width() // 3), 
                                     fill="white", 
                                     tags="button_icon")
        
        # If connected to a worker/supervisor instance, end that call too
        if app.connected_to and app.connected_to.root.winfo_exists():
            # Only trigger if the connected app hasn't already ended
            if app.connected_to.call_active:
                CallHandlers._terminate_call(app.connected_to)
                
        # Final cleanup
        app.connected_to = None
        app.is_ending_call = False
        
        # Update status text
        if MessageHandlers.preserve_history:
            app.status_label.config(text=f"{MessageHandlers.SAVE_EMOJI} History: Preserved")
        else:
            app.status_label.config(text="Ready for new call")
            
        # Toast notification to show call ended
        app.root.after(100, lambda: CallHandlers._show_call_ended_toast(app))

    @staticmethod
    def _show_call_ended_toast(app):
        """Show a toast notification for call ended."""
        if not hasattr(app, 'root') or not app.root.winfo_exists():
            return
            
        # Create a toast frame
        toast = tk.Toplevel(app.root)
        toast.overrideredirect(True)  # Remove window decorations
        toast.config(bg="#333333")
        toast.attributes("-topmost", True)
        
        # Position at the bottom of the app
        x = app.root.winfo_x() + app.root.winfo_width() // 2 - 150
        y = app.root.winfo_y() + app.root.winfo_height() - 100
        toast.geometry(f"300x60+{x}+{y}")
        
        # Add rounded corners using a canvas
        canvas = tk.Canvas(toast, bg="#333333", bd=0, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Call ended message
        toast_msg = tk.Label(
            canvas, 
            text=f"{MessageHandlers.CALL_EMOJI} Call ended",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#333333"
        )
        toast_msg.pack(pady=(10, 5))
        
        # Add duration if available
        if hasattr(app, 'call_duration'):
            minutes = app.call_duration // 60
            seconds = app.call_duration % 60
            duration_msg = tk.Label(
                canvas,
                text=f"Duration: {minutes:02d}:{seconds:02d}",
                font=("Segoe UI", 10),
                fg="#BBBBBB",
                bg="#333333"
            )
            duration_msg.pack()
        
        # Auto-close after 3 seconds
        toast.after(3000, toast.destroy)

    @staticmethod
    def _terminate_call(app):
        """Clean up after a call ends."""
        text_input = app.text_entry.get("1.0", tk.END)
        messagebox.showinfo("Call Complete", f"Voice call ended!\nMessage: {text_input}")

        app.text_entry.config(state=tk.NORMAL)
        app.text_entry.delete("1.0", tk.END)

        app.call_button_frame.destroy()
        from ui_components import UIComponents
        app.call_button_frame = UIComponents.create_circular_button(
            app,
            app.buttons_frame, 
            80, 
            "#00BFA5", 
            "Call", 
            command=app.toggle_call
        )
        app.call_button_frame.pack(side=tk.LEFT, padx=15)

    @staticmethod
    def update_duration(app):
        """Update the call duration display."""
        if not app.call_active or not hasattr(app, 'duration_label'):
            return
            
        app.call_duration += 1
        minutes = app.call_duration // 60
        seconds = app.call_duration % 60
        
        # Format with leading zeros for better readability
        app.duration_label.config(text=f"{minutes:02d}:{seconds:02d}")
        
        # Pulse the clock icon if available
        if app.call_duration % 2 == 0 and hasattr(app, 'duration_label'):
            if app.duration_label.winfo_parent():
                parent = app.duration_label.nametowidget(app.duration_label.winfo_parent())
                for widget in parent.winfo_children():
                    if isinstance(widget, tk.Canvas):
                        # Change color briefly to indicate active time
                        try:
                            # Find the clock's oval and hands by index rather than using "all"
                            # Typically the first item is the oval (clock face)
                            widget.itemconfig(1, outline="#00BFA5")  # Clock face
                            widget.itemconfig(2, fill="#00BFA5")     # Hour hand
                            widget.itemconfig(3, fill="#00BFA5")     # Minute hand
                            
                            # Schedule reverting to original color
                            def revert_color():
                                try:
                                    widget.itemconfig(1, outline="#8696A0")  # Clock face
                                    widget.itemconfig(2, fill="#8696A0")      # Hour hand
                                    widget.itemconfig(3, fill="#8696A0")      # Minute hand
                                except:
                                    pass  # In case widget is destroyed
                                    
                            app.root.after(500, revert_color)
                        except:
                            pass  # Ignore errors if widget structure is different
        
        # Continue updating every second
        app.root.after(1000, lambda: CallHandlers.update_duration(app)) 

    @staticmethod
    def save_conversation_to_text_file():
        """Save the full conversation to a text file."""
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_conversation_{timestamp}.txt"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("FULL CONVERSATION HISTORY\n")
                f.write("=" * 80 + "\n\n")
                
                # Write messages in chronological order with proper formatting
                for i in range(max(len(CallHandlers.worker_conversation), len(CallHandlers.supervisor_conversation))):
                    # Write worker message if available
                    if i < len(CallHandlers.worker_conversation):
                        f.write(f"\n👨‍💻 Worker says:\n{CallHandlers.worker_conversation[i]}\n\n")
                        f.write("-" * 40 + "\n")
                    
                    # Write supervisor message if available
                    if i < len(CallHandlers.supervisor_conversation):
                        f.write(f"\n👩‍💼 Supervisor says:\n{CallHandlers.supervisor_conversation[i]}\n\n")
                        f.write("-" * 40 + "\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"\n[SYSTEM] Conversation saved to file: {filename}\n")
            return filename
        except Exception as e:
            print(f"\n[SYSTEM] Error saving conversation to file: {e}\n")
            return None 