import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import threading
import io
from contextlib import redirect_stdout
from message_handlers import MessageHandlers
import time
import sys
import traceback

from redirector import StdoutRedirector

class CallHandlers:
    @staticmethod
    def toggle_call(app, event=None):
        """Toggle between starting and ending a call."""
        if not app.call_active:
            CallHandlers.start_call(app)
        else:
            CallHandlers.end_call(app)

    @staticmethod
    def start_call(app):
        """Start a call with another instance."""
        # Import here to avoid circular import
        from voice_call_app import VoiceCallApp
        from message_handlers import MessageHandlers
        
        # Ensure app is registered in instances list
        if app not in VoiceCallApp.instances:
            VoiceCallApp.instances.append(app)
            
        # Check if another call is already active
        if VoiceCallApp.active_call and VoiceCallApp.active_call != app:
            if VoiceCallApp.active_call.call_active:
                # Another call is active, try to connect to it
                other_app = VoiceCallApp.active_call
                
                # Connect the two apps
                app.connected_to = other_app
                other_app.connected_to = app
                
                # Establish call UI on both ends
                CallHandlers._establish_call(app, other_app)
                return
        
        # Get the input text from the text entry
        input_text = app.text_entry.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Input Required", "Please enter some text before calling.")
            return
            
        # Clear the input area
        app.text_entry.delete("1.0", tk.END)
        
        # Show thinking state in both apps
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
                                # Then update with the new response text after a delay
                                worker_app.root.after(1000, lambda w=worker_app, t=display_text: 
                                    self.safe_call(w, lambda: MessageHandlers._update_response_text(w, t, "worker_message")))
                                
                                if is_question and supervisor_app and widget_exists(supervisor_app.root):
                                    worker_app.root.after(5500, lambda w=worker_app, 
                                        t=f"{display_text}\n\nWaiting for Supervisor to respond...": 
                                        self.safe_call(w, lambda: MessageHandlers._update_response_text(w, t, "worker_message")))
                                    worker_app.root.after(6000, lambda s=supervisor_app: 
                                        self.safe_call(s, lambda: MessageHandlers.show_thinking_in_response(s, True)))
                            
                            # Update Supervisor app - send with "Worker:" prefix
                            if supervisor_app and widget_exists(supervisor_app.root):
                                # First, update the thinking state only (don't clear content)
                                supervisor_app.root.after(100, lambda s=supervisor_app: 
                                    self.safe_call(s, lambda: self._update_thinking_state_only(s, False)))
                                # Then update with the new response text after a delay
                                supervisor_app.root.after(1000, lambda s=supervisor_app, t=f"Worker: {display_text}": 
                                    self.safe_call(s, lambda: MessageHandlers._update_response_text(s, t, "worker_message")))
                                
                                if is_question:
                                    supervisor_app.root.after(5500, lambda s=supervisor_app: 
                                        self.safe_call(s, lambda: MessageHandlers.show_thinking_in_response(s, True)))
                            
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
                                # Then update with the new response text after a delay
                                supervisor_app.root.after(1000, lambda s=supervisor_app, t=display_text: 
                                    self.safe_call(s, lambda: MessageHandlers._update_response_text(s, t, "supervisor_message")))
                                
                                if is_question and worker_app and widget_exists(worker_app.root):
                                    supervisor_app.root.after(5500, lambda s=supervisor_app, 
                                        t=f"{display_text}\n\nWaiting for Worker to respond...": 
                                        self.safe_call(s, lambda: MessageHandlers._update_response_text(s, t, "supervisor_message")))
                                    supervisor_app.root.after(6000, lambda w=worker_app: 
                                        self.safe_call(w, lambda: MessageHandlers.show_thinking_in_response(w, True)))
                            
                            # Update Worker app - send with "Supervisor:" prefix
                            if worker_app and widget_exists(worker_app.root):
                                # First, update the thinking state only (don't clear content)
                                worker_app.root.after(100, lambda w=worker_app: 
                                    self.safe_call(w, lambda: self._update_thinking_state_only(w, False)))
                                # Then update with the new response text after a delay
                                worker_app.root.after(1000, lambda w=worker_app, t=f"Supervisor: {display_text}": 
                                    self.safe_call(w, lambda: MessageHandlers._update_response_text(w, t, "supervisor_message")))
                                
                                if is_question:
                                    worker_app.root.after(5500, lambda w=worker_app: 
                                        self.safe_call(w, lambda: MessageHandlers.show_thinking_in_response(w, True)))
                            
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
                stdout_redirector = DirectUIUpdatingRedirector(app.response_text, app)
                with redirect_stdout(stdout_redirector):
                    # Process the input in the background thread
                    from main import main
                    main(input_text)
                
                # Always force both apps to exit thinking state after processing
                app.root.after(0, lambda: MessageHandlers._force_exit_thinking_state(app))
                if app.connected_to:
                    app.connected_to.root.after(0, lambda: MessageHandlers._force_exit_thinking_state(app.connected_to))
            
            except Exception as e:
                # Handle errors and update UI from the main thread
                error_message = str(e)
                app.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {error_message}"))
                app.root.after(0, lambda: MessageHandlers.append_to_response(app, f"Error: {error_message}"))
                app.root.after(0, lambda: MessageHandlers._force_exit_thinking_state(app))
                if app.connected_to:
                    app.connected_to.root.after(0, lambda: MessageHandlers._force_exit_thinking_state(app.connected_to))

        # Create and start the thread
        threading_thread = threading.Thread(target=run_main_in_thread)
        threading_thread.daemon = True  # Thread will exit when main program exits
        threading_thread.start()
        
        # Keep the UI responsive by processing events
        app.root.update_idletasks()
        
        # Check if there's already an active call
        if (app.__class__.active_call is not None and 
            app.__class__.active_call != app and 
            app.__class__.active_call != app.connected_to):
            messagebox.showerror("Error", "Another call is already in progress!")
            return
            
        # Set this instance as the waiting call if no other call is waiting
        if app.__class__.waiting_call is None:
            app.__class__.waiting_call = app
            app.status_label.config(text="Waiting for other party to accept...")
            app.call_active = True
            app.start_call_button.pack_forget()
            app.stop_call_button.pack(pady=10)
            app.text_entry.config(state=tk.DISABLED)
            
            # Find another available instance and automatically connect
            for other_app in app.__class__.instances:
                if other_app != app and not other_app.call_active:
                    # Auto-accept call on the other instance
                    other_app.text_entry.insert("1.0", f"Auto-accepted call {other_app.model_path}\n")
                    other_app.connected_to = app
                    app.connected_to = other_app
                    app.__class__.active_call = (app, other_app)
                    
                    # Start the call for both apps
                    CallHandlers._establish_call(app, other_app)
                    CallHandlers._establish_call(other_app, app)
                    
                    app.__class__.waiting_call = None
        else:
            # Connect the calls if there's a waiting call
            other_app = app.__class__.waiting_call
            app.__class__.active_call = (other_app, app)
            
            # Setup both apps for the call
            app.connected_to = other_app
            other_app.connected_to = app
            
            # Start the call for both apps
            CallHandlers._establish_call(app, other_app)
            CallHandlers._establish_call(other_app, app)
            
            app.__class__.waiting_call = None

    @staticmethod
    def _establish_call(app, other_app):
        """Set up the UI for an established call."""
        from message_handlers import MessageHandlers
        
        app.call_active = True
        app.call_start_time = datetime.now()
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
        """End the current active call."""
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
        if connected_app and connected_app.root.winfo_exists():
            connected_app.status_label.config(text="Call ending...")
        
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
        
        # Reset the call state in both apps
        app.root.after(200, lambda: CallHandlers._complete_end_call(app))
        
        # If there's an active call with a connected app, also end it
        if connected_app and connected_app.root.winfo_exists():
            connected_app.root.after(300, lambda: CallHandlers._complete_end_call(connected_app))

    @staticmethod
    def _complete_end_call(app):
        """Complete the end call process."""
        # Import here to avoid circular import
        from voice_call_app import VoiceCallApp
        from message_handlers import MessageHandlers
        
        # Reset the call flags
        app.call_active = False
        app.call_start_time = None
        app.connected_to = None
        
        # Remove from active call tracking
        VoiceCallApp.active_call = None
        
        # Reset UI elements
        app.duration_label.pack_forget()
        app.duration_label.config(text="00:00")
        app.stop_call_button.pack_forget()
        app.start_call_button.pack(pady=10)
        
        # Manual disconnect from the call session
        CallHandlers._terminate_call(app)
        
        # Reset response areas
        app.status_label.config(text="Call ended")
        
        # Reset thinking state
        app.is_thinking = False
        
        # Clear the response area and show the call ended message
        app.response_text.config(state=tk.NORMAL)
        
        # First remove any thinking indicators to prevent them from persisting
        MessageHandlers._remove_thinking_indicators(app)
        
        # Use the safe clear text method to respect history preservation
        if MessageHandlers.preserve_history:
            # Add a call end marker to the conversation
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
            app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.CALL_EMOJI} Call ended. ", "system_header")
            app.response_text.insert(tk.END, "Enter a new message and press Call to start a new conversation.\n", "waiting_message")
            app.response_text.insert(tk.END, "═" * 60 + "\n\n", "separator")
        else:
            # If not preserving history, clear everything and show a simple message
            MessageHandlers.safe_clear_text(app)
            app.response_text.insert(tk.END, f"{MessageHandlers.CALL_EMOJI} Call ended. Enter a new message and press Call to start a new conversation.", "supervisor_message")
        
        # Ensure the message is visible with improved scrolling
        MessageHandlers._ensure_autoscroll(app)
        app.response_text.config(state=tk.DISABLED)

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
        if app.call_active and app.call_start_time:
            now = datetime.now()
            diff = now - app.call_start_time
            minutes = diff.seconds // 60
            seconds = diff.seconds % 60
            call_duration = f"{minutes:02d}:{seconds:02d}"
            app.duration_label.config(text=call_duration)
            app.root.after(1000, lambda: CallHandlers.update_duration(app)) 