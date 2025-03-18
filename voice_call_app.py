import tkinter as tk
import os

from ui_components import UIComponents
from message_handlers import MessageHandlers
from call_handlers import CallHandlers
from minion_terminal import MinionTerminal, TerminalTheme

class VoiceCallApp:
    # Call state tracking
    active_call = None
    waiting_call = None
    # Shared minion terminal instance
    minion_terminal = None
    minion_window = None
    # Track all instances for communication
    instances = []
    # Theme colors for consistent UI
    THEME_COLORS = {
        "bg": "#121B22",
        "panel": "#1F2C34",
        "text": "#E9EDEF",
        "accent": "#00A884",
        "inactive": "#8696A0",
        "message_sent": "#005C4B",
        "message_received": "#1F2C34"
    }

    def __init__(self, root, model_name, model_path):
        """Initialize the VoiceCallApp with the given parameters."""
        self.root = root
        self.model_name = model_name
        self.model_path = model_path
        
        # Set model label based on model name
        if "worker" in model_name.lower():
            self.model_label = "Worker (Local)"
        elif "supervisor" in model_name.lower():
            self.model_label = "Supervisor (Remote)"
        else:
            self.model_label = model_name
            
        # Initialize the minion terminal only once for the first instance
        if VoiceCallApp.minion_terminal is None:
            VoiceCallApp.minion_window = tk.Toplevel(root)
            VoiceCallApp.minion_window.title("Minions Terminal")
            VoiceCallApp.minion_window.iconify()  # Start minimized
            VoiceCallApp.minion_terminal = MinionTerminal(VoiceCallApp.minion_window)
        
        # Initialize state variables
        self.call_active = False
        self.call_start_time = None
        self.duration_timer = None
        self.connected_to = None
        self.is_thinking = False
        self.is_ending_call = False
        self._thinking_after_id = None
        self._pending_replace_id = None
        self._pending_message = None
        self._pending_tag = None
        
        # Setup UI components
        self.setup_ui()
        
        # Ensure conversation history directory exists
        os.makedirs(MessageHandlers.conversation_history_dir, exist_ok=True)
        
        # Add this instance to the instances list for tracking
        VoiceCallApp.instances.append(self)
        CallHandlers.instances.append(self)
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def on_close(self):
        """Handle window close event."""
        # End any active call
        if self.call_active:
            self.end_call()
            
        # Remove this instance from tracking lists
        if self in VoiceCallApp.instances:
            VoiceCallApp.instances.remove(self)
        if self in CallHandlers.instances:
            CallHandlers.instances.remove(self)
            
        # If this is the last window, also close the minion terminal
        if not VoiceCallApp.instances and VoiceCallApp.minion_window:
            VoiceCallApp.minion_window.destroy()
            
        # Destroy this window
        self.root.destroy()
    
    def setup_ui(self):
        """Set up the UI for the app with modern design elements."""
        UIComponents.setup_ui(self)
        
        # Add tooltip to call button
        if hasattr(self, 'call_btn'):
            self.create_tooltip(self.call_btn, "Start or end a voice call")
            
        # Add tooltip to send minion button
        if hasattr(self, 'send_minion_btn'):
            self.create_tooltip(self.send_minion_btn, "Send message to Minion terminal")
            
        # Add tooltip to history button
        if hasattr(self, 'history_btn'):
            self.create_tooltip(self.history_btn, "Toggle conversation history")
            
        # Add animation to profile avatar
        if hasattr(self, 'avatar_canvas'):
            self.animate_avatar()
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        tooltip = tk.Toplevel(self.root)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        tooltip.attributes("-topmost", True)
        
        # Create tooltip content
        padding = 4
        label = tk.Label(
            tooltip, 
            text=text, 
            bg="#333333", 
            fg="white",
            padx=padding,
            pady=padding,
            wraplength=200,
            font=("Segoe UI", 9)
        )
        label.pack()
        
        def show_tooltip(event):
            x = widget.winfo_rootx() + widget.winfo_width() // 2
            y = widget.winfo_rooty() + widget.winfo_height() + 5
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()
            
        def hide_tooltip(event):
            tooltip.withdraw()
            
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
    
    def animate_avatar(self):
        """Add subtle animation to the avatar."""
        if not hasattr(self, 'avatar_canvas') or not self.avatar_canvas.winfo_exists():
            return
            
        # Glow animation cycle
        def glow_cycle(step=0, direction=1):
            if not hasattr(self, 'avatar_canvas') or not self.avatar_canvas.winfo_exists():
                return
                
            # Change outline width to create subtle pulsing effect
            if self.call_active:
                width = 2 + (step / 5)
                self.avatar_canvas.delete("avatar_glow")
                self.avatar_canvas.create_oval(
                    2, 2, 58, 58, 
                    outline=UIComponents.HIGHLIGHT_COLOR, 
                    width=width,
                    tags="avatar_glow"
                )
                
                # Update step for next cycle
                if step >= 5:
                    direction = -1
                elif step <= 0:
                    direction = 1
                    
                new_step = step + (0.5 * direction)
                self.root.after(100, lambda: glow_cycle(new_step, direction))
                
        # Start animation if call is active
        if self.call_active:
            glow_cycle()
    
    # Delegate call-related functions to CallHandlers
    def toggle_call(self, event=None):
        """Toggle between starting and ending a call."""
        CallHandlers.toggle_call(self, event)
    
    def start_call(self):
        """Start a call with another instance."""
        CallHandlers.start_call(self)
        # Start avatar animation when call starts
        self.animate_avatar()
    
    def end_call(self):
        """End the current active call."""
        CallHandlers.end_call(self)
    
    # Delegate message-related functions to MessageHandlers
    def clear_response(self):
        """Clear the response text area."""
        MessageHandlers.clear_response(self)
    
    def append_to_response(self, text):
        """Append text to the response area with enhanced formatting."""
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
        """Toggle or set whether to preserve conversation history."""
        new_value = MessageHandlers.toggle_preserve_history(self, preserve)
        
        # Update history button appearance based on new value
        if hasattr(self, 'history_btn'):
            color = UIComponents.MESSAGE_BG_RECEIVED if new_value else UIComponents.INACTIVE_COLOR
            # Update button color
            if hasattr(self.history_btn, 'itemconfig'):
                self.history_btn.itemconfig(1, fill=color)  # Update oval fill color
                
        return new_value
    
    def send_to_minion(self):
        """Send the current text entry message to the minion terminal."""
        UIComponents.send_to_minion_terminal(self)
        
    def show_notification(self, title, message, duration=3000):
        """Show a notification toast with the given title and message."""
        # Create toast frame
        toast = tk.Toplevel(self.root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(bg="#333333")
        
        # Position at the bottom of the window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + self.root.winfo_height() - 100
        toast.geometry(f"300x80+{x}+{y}")
        
        # Add title with icon
        title_frame = tk.Frame(toast, bg="#333333")
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        icon_label = tk.Label(
            title_frame, 
            text=MessageHandlers.INFO_ICON,
            font=("Segoe UI Emoji", 14),
            fg="#00A884",
            bg="#333333"
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 5))
        
        title_label = tk.Label(
            title_frame,
            text=title,
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg="#333333",
            anchor=tk.W
        )
        title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add message
        message_label = tk.Label(
            toast,
            text=message,
            font=("Segoe UI", 10),
            fg="#CCCCCC",
            bg="#333333",
            wraplength=280,
            justify=tk.LEFT,
            anchor=tk.W
        )
        message_label.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Add close button
        close_button = tk.Label(
            title_frame,
            text="✕",
            font=("Segoe UI", 12),
            fg="#999999",
            bg="#333333"
        )
        close_button.pack(side=tk.RIGHT)
        
        # Bind close button
        close_button.bind("<Button-1>", lambda e: toast.destroy())
        
        # Add hover effect to close button
        def on_close_enter(e):
            close_button.config(fg="white")
            
        def on_close_leave(e):
            close_button.config(fg="#999999")
            
        close_button.bind("<Enter>", on_close_enter)
        close_button.bind("<Leave>", on_close_leave)
        
        # Auto-close after specified duration
        toast.after(duration, toast.destroy)
        
        return toast 