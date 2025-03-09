import tkinter as tk
import os

from ui_components import UIComponents
from message_handlers import MessageHandlers
from call_handlers import CallHandlers
from minion_terminal import MinionTerminal

class VoiceCallApp:
    # Call state tracking
    active_call = None
    waiting_call = None
    # Shared minion terminal instance
    minion_terminal = None
    minion_window = None

    def __init__(self, root, model_name, model_path):
        """Initialize the VoiceCallApp with the given parameters."""
        self.root = root
        self.model_name = model_name
        self.model_path = model_path
        self.model_label = "Model"  # Default label, will be overridden
        
        # Initialize the minion terminal only once for the first instance
        if VoiceCallApp.minion_terminal is None:
            VoiceCallApp.minion_window = tk.Toplevel(root)
            VoiceCallApp.minion_window.title("Minions Terminal")
            VoiceCallApp.minion_terminal = MinionTerminal(VoiceCallApp.minion_window)
        
        self.setup_ui()
        self.call_active = False
        self.call_start_time = None
        self.duration_timer = None
        self.connected_to = None
        self.is_thinking = False
        self.is_ending_call = False  # Track if call is in the process of ending
        self._thinking_after_id = None  # Initialize the thinking animation timer ID
        self._pending_replace_id = None  # Track pending replacements
        self._pending_message = None  # Store pending message for delayed display
        self._pending_tag = None  # Store pending tag for delayed display
        
        # Ensure conversation history directory exists
        os.makedirs(MessageHandlers.conversation_history_dir, exist_ok=True)
        
        # Add this instance to the CallHandlers instances list
        CallHandlers.instances.append(self)
    
    def setup_ui(self):
        """Set up the UI for the app."""
        UIComponents.setup_ui(self)
    
    # Delegate call-related functions to CallHandlers
    def toggle_call(self, event=None):
        """Toggle between starting and ending a call."""
        CallHandlers.toggle_call(self, event)
    
    def start_call(self):
        """Start a call with another instance."""
        CallHandlers.start_call(self)
    
    def end_call(self):
        """End the current active call."""
        CallHandlers.end_call(self)
    
    # Delegate message-related functions to MessageHandlers
    def clear_response(self):
        """Clear the response text area."""
        MessageHandlers.clear_response(self)
    
    def append_to_response(self, text):
        """Append text to the response area."""
        MessageHandlers.append_to_response(self, text)
    
    def show_thinking_in_response(self, is_thinking=True):
        """Shows or hides the thinking indicator in the response area."""
        MessageHandlers.show_thinking_in_response(self, is_thinking)
    
    def update_thinking_status(self):
        """Update the status label to show thinking animation."""
        MessageHandlers.update_thinking_status(self)
    
    def _delayed_thinking_update(self):
        """Handle delayed update for thinking state."""
        MessageHandlers._delayed_thinking_update(self)
    
    def _force_exit_thinking_state_if_still_thinking(self):
        """Force exit thinking state only if still in thinking state."""
        MessageHandlers._force_exit_thinking_state_if_still_thinking(self)
    
    def _force_exit_thinking_state(self):
        """Force exit thinking state."""
        MessageHandlers._force_exit_thinking_state(self)
    
    def _update_response_text(self, text, tag):
        """Update the response text with the given text and tag."""
        MessageHandlers._update_response_text(self, text, tag)
    
    def _delayed_text_update(self, text, tag):
        """Handle delayed text update."""
        MessageHandlers._delayed_text_update(self, text, tag)
        
    def toggle_preserve_history(self, preserve=None):
        """Toggle or set whether to preserve conversation history.
        
        Args:
            preserve: If provided, sets the preserve_history flag to this value.
                     If None, toggles the current value.
        
        Returns:
            The new preserve_history value
        """
        return MessageHandlers.toggle_preserve_history(self, preserve)
    
    def send_to_minion(self):
        """Send the current text entry message to the minion terminal"""
        # DEPRECATED: This method is kept for backward compatibility.
        # UI components now communicate directly with minion_terminal
        
        # Make sure minion terminal exists
        if VoiceCallApp.minion_terminal is None:
            return
            
        message = self.text_entry.get("1.0", "end-1c").strip()
        if message and message != "Type a message...":
            # Set the message in minion_terminal
            VoiceCallApp.minion_terminal.text_entry.delete("1.0", tk.END)
            VoiceCallApp.minion_terminal.text_entry.insert("1.0", message)
            
            # Send the message directly to minion_terminal
            VoiceCallApp.minion_terminal.send_message()
            
            # Clear the input field after sending
            self.text_entry.delete("1.0", tk.END)
            self.text_entry.insert("1.0", "Type a message...")
            self.text_entry.configure(fg="#8696A0")  # Reset to placeholder color
            
            # Bring minion terminal window to front
            if VoiceCallApp.minion_window:
                VoiceCallApp.minion_window.lift() 