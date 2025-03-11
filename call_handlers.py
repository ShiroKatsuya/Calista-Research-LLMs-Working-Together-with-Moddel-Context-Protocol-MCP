import tkinter as tk
from tkinter import messagebox
import sys
import threading
import traceback
import time
from datetime import datetime
from contextlib import redirect_stdout
# Don't import VoiceCallApp directly to avoid circular import
from message_handlers import MessageHandlers
from redirector import StdoutRedirector
from ui_components import UIComponents
import os
import math

class CallHandlers:
    # Class-level list to track app instances (moved from VoiceCallApp)
    instances = []
    
    # Track conversation content for terminal display
    worker_conversation = []
    supervisor_conversation = []
    
    # Animation constants
    PULSE_DURATION = 50  # ms between animation frames
    PULSE_CYCLES = 10    # number of cycles for pulse animation
    TYPING_DELAY = 50    # ms between characters when typing animation
    
    @staticmethod
    def print_to_terminal(message, sender=None, is_system=False):
        """Print a message to the terminal with proper formatting."""
        # Setup message formatting based on sender
        if is_system:
            prefix = f"{MessageHandlers.SYSTEM_EMOJI} System: "
            tag = "system_message"
            # Add to both conversation histories
            CallHandlers.worker_conversation.append(f"[SYSTEM] {message}")
            CallHandlers.supervisor_conversation.append(f"[SYSTEM] {message}")
        elif sender == "worker":
            prefix = f"{MessageHandlers.WORKER_EMOJI} Worker: "
            tag = "worker_message"
            # Add to worker conversation
            CallHandlers.worker_conversation.append(f"[WORKER] {message}")
        elif sender == "supervisor":
            prefix = f"{MessageHandlers.SUPERVISOR_EMOJI} Supervisor: "
            tag = "supervisor_message"
            # Add to supervisor conversation
            CallHandlers.supervisor_conversation.append(f"[SUPERVISOR] {message}")
        else:
            prefix = ""
            tag = ""
            
        # Timestamp the message
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {prefix}{message}\n"
        
        # Print to minion terminal if exists
        for app in CallHandlers.instances:
            if hasattr(app, 'minion_terminal') and app.minion_terminal is not None:
                app.minion_terminal.output_text.configure(state=tk.NORMAL)
                app.minion_terminal.output_text.insert(tk.END, formatted_message, tag)
                app.minion_terminal.output_text.see(tk.END)
                app.minion_terminal.output_text.configure(state=tk.DISABLED)
    
    @staticmethod
    def print_full_conversation():
        """Print the full conversation history to the terminal."""
        # Create a visual separator
        separator = f"\n{'-' * 50}\n"
        
        # Define header with timestamp and emojis
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"{MessageHandlers.SYSTEM_EMOJI} CONVERSATION HISTORY ({timestamp}) {MessageHandlers.SYSTEM_EMOJI}\n"
        
        # Print worker conversation
        worker_header = f"{MessageHandlers.WORKER_EMOJI} WORKER CONVERSATION {MessageHandlers.WORKER_EMOJI}\n"
        worker_content = "\n".join(CallHandlers.worker_conversation)
        
        # Print supervisor conversation  
        supervisor_header = f"{MessageHandlers.SUPERVISOR_EMOJI} SUPERVISOR CONVERSATION {MessageHandlers.SUPERVISOR_EMOJI}\n"
        supervisor_content = "\n".join(CallHandlers.supervisor_conversation)
        
        # Format the full output with visual elements
        full_output = (
            f"{separator}{header}{separator}"
            f"{worker_header}{separator}{worker_content}\n{separator}"
            f"{supervisor_header}{separator}{supervisor_content}\n{separator}"
        )
        
        # Print to minion terminal if it exists
        for app in CallHandlers.instances:
            if hasattr(app, 'minion_terminal') and app.minion_terminal is not None:
                app.minion_terminal.output_text.configure(state=tk.NORMAL)
                app.minion_terminal.output_text.insert(tk.END, full_output)
                app.minion_terminal.output_text.see(tk.END)
                app.minion_terminal.output_text.configure(state=tk.DISABLED)
                
                # Update status to show completion
                app.minion_terminal.update_status("Conversation history displayed", "#3498db")
    
    @staticmethod
    def toggle_call(app, event=None):
        """Toggle between starting and ending a call."""
        if app.call_active:
            CallHandlers.end_call(app)
        else:
            CallHandlers.start_call(app)
    
    @staticmethod
    def start_call(app):
        """Start a call with another instance."""
        # Clear previous conversations
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
            
            # Find the other app instance to call
            other_app = None
            for instance in app.__class__.instances:
                if instance != app:
                    other_app = instance
                    break
            
            if other_app is None:
                messagebox.showerror("Error", "No other app instance found to call.")
                return
            
            # Show calling interface with WhatsApp-style ringing animation
            CallHandlers._show_calling_interface(app, other_app, input_text)
            
            # The rest of the logic will continue in _show_calling_interface
        else:
            messagebox.showerror("Error", "Text entry not found")
    
    @staticmethod
    def _show_calling_interface(app, other_app, input_text):
        """Display a WhatsApp-style calling interface with ringing animation."""
        # Save the input text for later processing
        app.pending_message = input_text
        
        # Create a calling overlay
        app.calling_overlay = tk.Toplevel(app.root)
        app.calling_overlay.title("Calling")
        app.calling_overlay.geometry("300x450")
        app.calling_overlay.configure(bg=UIComponents.DARK_BG)
        app.calling_overlay.transient(app.root)  # Make it appear related to main window
        
        # Center the overlay relative to the app window
        app.root.update_idletasks()
        x = app.root.winfo_rootx() + (app.root.winfo_width() // 2) - 150
        y = app.root.winfo_rooty() + (app.root.winfo_height() // 2) - 225
        app.calling_overlay.geometry(f"+{x}+{y}")
        
        # Prevent closing with X button
        app.calling_overlay.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Add caller avatar and name at the top
        avatar_frame = tk.Frame(app.calling_overlay, bg=UIComponents.DARK_BG)
        avatar_frame.pack(pady=(40, 10))
        
        # Large circular avatar
        avatar_size = 120
        avatar_canvas = tk.Canvas(avatar_frame, width=avatar_size, height=avatar_size, 
                               bg=UIComponents.DARK_BG, highlightthickness=0)
        avatar_canvas.pack()
        
        # Draw avatar circle
        avatar_canvas.create_oval(5, 5, avatar_size-5, avatar_size-5, 
                               fill=UIComponents.PANEL_BG, outline=UIComponents.HIGHLIGHT_COLOR, width=2)
        
        # Add initials in the center
        model_initials = other_app.model_label[0] if other_app.model_label else "S"
        avatar_canvas.create_text(avatar_size//2, avatar_size//2, 
                               text=model_initials, fill=UIComponents.TEXT_COLOR, 
                               font=("Segoe UI", 36, "bold"))
        
        # Name label
        name_label = tk.Label(app.calling_overlay, text=other_app.model_label,
                           font=("Segoe UI", 20, "bold"), fg=UIComponents.TEXT_COLOR, bg=UIComponents.DARK_BG)
        name_label.pack(pady=(10, 5))
        
        # Calling status label
        app.calling_status = tk.Label(app.calling_overlay, text="Calling...",
                                   font=("Segoe UI", 14), fg=UIComponents.INACTIVE_COLOR, bg=UIComponents.DARK_BG)
        app.calling_status.pack(pady=5)
        
        # Add ringing animation
        ring_canvas = tk.Canvas(app.calling_overlay, width=180, height=40, 
                             bg=UIComponents.DARK_BG, highlightthickness=0)
        ring_canvas.pack(pady=20)
        
        # Create wave bars for ringing animation
        wave_bars = []
        for i in range(5):
            bar = ring_canvas.create_rectangle(
                20 + i*30, 20, 40 + i*30, 20,
                fill=UIComponents.HIGHLIGHT_COLOR, width=0
            )
            wave_bars.append(bar)
        
        # End call button (red circle)
        button_frame = tk.Frame(app.calling_overlay, bg=UIComponents.DARK_BG)
        button_frame.pack(side=tk.BOTTOM, pady=40)
        
        end_call_btn = tk.Canvas(button_frame, width=70, height=70, 
                              bg=UIComponents.DARK_BG, highlightthickness=0)
        end_call_btn.pack()
        
        # Red circle for end call
        end_call_circle = end_call_btn.create_oval(5, 5, 65, 65, fill="#E53935", outline="")
        end_call_phone = end_call_btn.create_text(35, 35, text="📞", font=("Segoe UI", 24), fill="white")
        
        # Bind end call action
        end_call_btn.tag_bind(end_call_circle, "<Button-1>", 
                           lambda e: CallHandlers._cancel_outgoing_call(app))
        end_call_btn.tag_bind(end_call_phone, "<Button-1>", 
                           lambda e: CallHandlers._cancel_outgoing_call(app))
        
        # Animate the wave bars
        def animate_wave(step=0):
            if not hasattr(app, 'calling_overlay') or not app.calling_overlay.winfo_exists():
                return
                
            # Calculate heights for wave effect
            heights = [
                15 + 10 * abs(math.sin((step + i) / 2)),
                15 + 10 * abs(math.sin((step + i + 2) / 2)),
                15 + 10 * abs(math.sin((step + i + 4) / 2)),
                15 + 10 * abs(math.sin((step + i + 6) / 2)),
                15 + 10 * abs(math.sin((step + i + 8) / 2))
            ]
            
            # Update each bar
            for i, bar in enumerate(wave_bars):
                ring_canvas.coords(bar, 20 + i*30, 40 - heights[i], 40 + i*30, 40)
            
            # Continue animation
            app.calling_overlay.after(100, lambda: animate_wave(step + 1))
        
        # Start wave animation
        animate_wave()
        
        # Auto-accept call in other app after delay (simulating WhatsApp behavior)
        app.calling_overlay.after(2000, lambda: CallHandlers._auto_accept_call(other_app, app))

    @staticmethod
    def _cancel_outgoing_call(app):
        """Cancel an outgoing call attempt."""
        if hasattr(app, 'calling_overlay') and app.calling_overlay.winfo_exists():
            app.calling_overlay.destroy()
            delattr(app, 'calling_overlay')
            
        # Reset any call state
        if app.__class__.waiting_call:
            app.__class__.waiting_call = None
        
        # Notify the user
        app.status_label.config(text="Call canceled")
        
        # Log to terminal
        CallHandlers.print_to_terminal("Outgoing call canceled", is_system=True)

    @staticmethod
    def _auto_accept_call(receiving_app, calling_app):
        """Auto-accept an incoming call after a short delay."""
        # Check if the call was canceled
        if not hasattr(calling_app, 'calling_overlay') or not calling_app.calling_overlay.winfo_exists():
            return
            
        # Update the status text to show connecting
        calling_app.calling_status.config(text="Connecting...")
        
        # Show incoming call notification in receiving app
        CallHandlers._show_incoming_call(receiving_app, calling_app)
        
        # Auto accept after a delay (simulating auto-answer)
        receiving_app.root.after(2000, lambda: CallHandlers._accept_incoming_call(receiving_app, calling_app))

    @staticmethod
    def _show_incoming_call(receiving_app, calling_app):
        """Show an incoming call notification in WhatsApp style."""
        # Create incoming call notification
        receiving_app.incoming_overlay = tk.Toplevel(receiving_app.root)
        receiving_app.incoming_overlay.title("Incoming Call")
        receiving_app.incoming_overlay.geometry("300x450")
        receiving_app.incoming_overlay.configure(bg=UIComponents.DARK_BG)
        receiving_app.incoming_overlay.transient(receiving_app.root)
        
        # Center the overlay
        receiving_app.root.update_idletasks()
        x = receiving_app.root.winfo_rootx() + (receiving_app.root.winfo_width() // 2) - 150
        y = receiving_app.root.winfo_rooty() + (receiving_app.root.winfo_height() // 2) - 225
        receiving_app.incoming_overlay.geometry(f"+{x}+{y}")
        
        # Prevent closing with X button
        receiving_app.incoming_overlay.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Add caller info
        avatar_frame = tk.Frame(receiving_app.incoming_overlay, bg=UIComponents.DARK_BG)
        avatar_frame.pack(pady=(40, 10))
        
        # Avatar
        avatar_size = 120
        avatar_canvas = tk.Canvas(avatar_frame, width=avatar_size, height=avatar_size, 
                               bg=UIComponents.DARK_BG, highlightthickness=0)
        avatar_canvas.pack()
        
        # Draw avatar
        avatar_canvas.create_oval(5, 5, avatar_size-5, avatar_size-5, 
                               fill=UIComponents.PANEL_BG, outline=UIComponents.HIGHLIGHT_COLOR, width=2)
        
        # Add initials
        model_initials = calling_app.model_label[0] if calling_app.model_label else "W"
        avatar_canvas.create_text(avatar_size//2, avatar_size//2, 
                               text=model_initials, fill=UIComponents.TEXT_COLOR, 
                               font=("Segoe UI", 36, "bold"))
        
        # Name and status
        name_label = tk.Label(receiving_app.incoming_overlay, text=calling_app.model_label,
                           font=("Segoe UI", 20, "bold"), fg=UIComponents.TEXT_COLOR, bg=UIComponents.DARK_BG)
        name_label.pack(pady=(10, 5))
        
        status = tk.Label(receiving_app.incoming_overlay, text="Incoming voice call...",
                       font=("Segoe UI", 14), fg=UIComponents.INACTIVE_COLOR, bg=UIComponents.DARK_BG)
        status.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(receiving_app.incoming_overlay, bg=UIComponents.DARK_BG)
        button_frame.pack(side=tk.BOTTOM, pady=40)
        
        # Accept call button (green)
        accept_btn = tk.Canvas(button_frame, width=70, height=70, 
                            bg=UIComponents.DARK_BG, highlightthickness=0)
        accept_btn.pack(side=tk.LEFT, padx=20)
        
        accept_circle = accept_btn.create_oval(5, 5, 65, 65, fill="#4CAF50", outline="")
        accept_phone = accept_btn.create_text(35, 35, text="📞", font=("Segoe UI", 24), fill="white")
        
        # Reject call button (red)
        reject_btn = tk.Canvas(button_frame, width=70, height=70, 
                            bg=UIComponents.DARK_BG, highlightthickness=0)
        reject_btn.pack(side=tk.RIGHT, padx=20)
        
        reject_circle = reject_btn.create_oval(5, 5, 65, 65, fill="#E53935", outline="")
        reject_phone = reject_btn.create_text(35, 35, text="📞", font=("Segoe UI", 24), fill="white", angle=135)
        
        # Bind actions
        accept_btn.tag_bind(accept_circle, "<Button-1>", 
                         lambda e: CallHandlers._accept_incoming_call(receiving_app, calling_app))
        accept_btn.tag_bind(accept_phone, "<Button-1>", 
                         lambda e: CallHandlers._accept_incoming_call(receiving_app, calling_app))
        
        reject_btn.tag_bind(reject_circle, "<Button-1>", 
                         lambda e: CallHandlers._reject_incoming_call(receiving_app, calling_app))
        reject_btn.tag_bind(reject_phone, "<Button-1>", 
                         lambda e: CallHandlers._reject_incoming_call(receiving_app, calling_app))
        
        # Play ringing sound (if sound available)
        # This would need to be implemented with a sound library

    @staticmethod
    def _accept_incoming_call(receiving_app, calling_app):
        """Accept an incoming call and establish the connection."""
        # Remove incoming call overlay
        if hasattr(receiving_app, 'incoming_overlay') and receiving_app.incoming_overlay.winfo_exists():
            receiving_app.incoming_overlay.destroy()
            delattr(receiving_app, 'incoming_overlay')
        
        # Remove calling overlay
        if hasattr(calling_app, 'calling_overlay') and calling_app.calling_overlay.winfo_exists():
            calling_app.calling_overlay.destroy()
            delattr(calling_app, 'calling_overlay')
        
        # Get the pending message from the calling app
        input_text = calling_app.pending_message if hasattr(calling_app, 'pending_message') else ""
        
        # Establish the call connection
        CallHandlers._establish_call(calling_app, receiving_app)
        
        # Send the pending message to minion_terminal
        if input_text and hasattr(calling_app.__class__, 'minion_terminal'):
            # Set the text entry in minion_terminal to this message
            calling_app.__class__.minion_terminal.text_entry.delete("1.0", tk.END)
            calling_app.__class__.minion_terminal.text_entry.insert("1.0", input_text)
            
            # Start the minion conversation
            calling_app.__class__.minion_terminal.start_minion_conversation()
            
            # Clear the input field after sending
            calling_app.text_entry.delete("1.0", tk.END)
            calling_app.text_entry.insert("1.0", "Type a message...")
            calling_app.text_entry.config(fg="#8696A0")

    @staticmethod
    def _reject_incoming_call(receiving_app, calling_app):
        """Reject an incoming call."""
        # Remove incoming call overlay
        if hasattr(receiving_app, 'incoming_overlay') and receiving_app.incoming_overlay.winfo_exists():
            receiving_app.incoming_overlay.destroy()
            delattr(receiving_app, 'incoming_overlay')
        
        # Update caller's UI to show rejected
        if hasattr(calling_app, 'calling_overlay') and calling_app.calling_overlay.winfo_exists():
            calling_app.calling_status.config(text="Call rejected")
            calling_app.calling_overlay.after(1500, calling_app.calling_overlay.destroy)
        
        # Log to terminal
        CallHandlers.print_to_terminal("Call rejected", is_system=True)

    @staticmethod
    def _establish_call(app, other_app):
        """Establish a call between two app instances."""
        # Make sure instances exist
        if app is None or other_app is None:
            return
            
        # Create visual connection animation between windows
        CallHandlers._create_connection_animation(app, other_app)
        
        # Update call state
        app.call_active = True
        app.call_start_time = time.time()
        app.connected_to = other_app
        
        # Set the active_call class variable through app's class
        app.__class__.active_call = (app, other_app)
        
        # Reset waiting call
        app.__class__.waiting_call = None
        
        # Configure UI to show active call with enhanced visual feedback
        app.status_label.config(text=f"Connected to {other_app.model_label}")
        
        # Change call button to a red "end call" button
        if hasattr(app, 'call_btn'):
            app.call_btn.itemconfig("button_bg", fill="#E53935")  # Red for end call
            app.call_btn.itemconfig("button_icon", text="🔴")  # Change icon
            
        # Add subtle glow effect to the avatar to show active status
        if hasattr(app, 'avatar_canvas'):
            for i in range(3):
                app.avatar_canvas.create_oval(
                    5 + i*2, 5 + i*2, 
                    55 - i*2, 55 - i*2, 
                    outline="#4CAF50",  # Green glow
                    width=1,
                    tags="active_call_glow"
                )
                
        # Highlight the name label with active color
        if hasattr(app, 'name_label'):
            app.name_label.config(fg="#4CAF50")  # Green for active
            
        # Make sure the text entry is clear and ready for input
        if hasattr(app, 'text_entry'):
            app.text_entry.delete("1.0", tk.END)
            
        # Clear the response area and add initial message
        app.clear_response()
        
        # Add animated typing greeting
        greeting = f"Connected to {other_app.model_label}. Call is active now."
        current_pos = 0
        
        def animate_greeting():
            nonlocal current_pos
            if current_pos <= len(greeting):
                app.append_to_response(greeting[:current_pos])
                current_pos += 3  # Add 3 characters at a time for speed
                app.root.after(50, animate_greeting)
                
        animate_greeting()
        
        # Start the call duration timer
        CallHandlers.update_duration(app)

    @staticmethod
    def _create_connection_animation(app1, app2):
        """Create a visual animation connecting the two call windows when a call is established."""
        # Create a transparent window for the animation
        animation_window = tk.Toplevel()
        animation_window.attributes("-alpha", 0.8)
        animation_window.attributes("-topmost", True)
        animation_window.overrideredirect(True)  # Remove window decorations
        
        # Get positions of both windows
        app1.root.update_idletasks()
        app2.root.update_idletasks()
        
        x1, y1 = app1.root.winfo_rootx() + app1.root.winfo_width()//2, app1.root.winfo_rooty() + app1.root.winfo_height()//2
        x2, y2 = app2.root.winfo_rootx() + app2.root.winfo_width()//2, app2.root.winfo_rooty() + app2.root.winfo_height()//2
        
        # Position and size the animation window to cover the space between windows
        width = abs(x2 - x1) + 100  # Add some padding
        height = abs(y2 - y1) + 100  # Add some padding
        
        # Calculate position to connect window centers
        start_x = min(x1, x2) - 50
        start_y = min(y1, y2) - 50
        
        # Create a window that encompasses both points with padding
        animation_window.geometry(f"{width}x{height}+{start_x}+{start_y}")
        
        # Create a canvas for the animation
        canvas = tk.Canvas(animation_window, bg=UIComponents.DARKER_BG, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Calculate the bezier curve to create a smooth arc between windows
        def bezier_point(t, p0, p1, p2):
            """Calculate point on a quadratic bezier curve."""
            return (1-t)**2 * p0 + 2 * (1-t) * t * p1 + t**2 * p2
        
        # Convert absolute coordinates to relative to animation window
        x1_rel = x1 - start_x
        y1_rel = y1 - start_y
        x2_rel = x2 - start_x
        y2_rel = y2 - start_y
        
        # Control point for the bezier curve (raise it above the direct line)
        mid_x = (x1_rel + x2_rel) / 2
        mid_y = min(y1_rel, y2_rel) - 70  # Raise the curve up
        
        # Generate points along the bezier curve
        curve_points = []
        for i in range(31):  # More points = smoother curve
            t = i / 30
            x = bezier_point(t, x1_rel, mid_x, x2_rel)
            y = bezier_point(t, y1_rel, mid_y, y2_rel)
            curve_points.append((x, y))
        
        # Animation function
        def animate_connection(step=0, max_steps=20):
            if step > max_steps:
                # Show connected animation
                points_flat = []
                for x, y in curve_points:
                    points_flat.extend([x, y])
                
                # Draw final connection line
                canvas.delete("all")
                connection_line = canvas.create_line(
                    points_flat,
                    width=4, 
                    fill=UIComponents.ANIMATION_COLOR_ACTIVE,
                    smooth=True,
                    dash=(10, 4),
                    dashoffset=step % 14
                )
                
                # Add connection glow
                for i in range(3):
                    canvas.create_line(
                        points_flat,
                        width=6 + i*3, 
                        fill=UIComponents.ANIMATION_COLOR_ACTIVE,
                        smooth=True,
                        dash=(10, 4),
                        dashoffset=step % 14,
                        stipple="gray50",  # Create glow effect
                        state=tk.DISABLED
                    )
                
                # Add connection text in the middle
                text_x = (x1_rel + x2_rel) / 2
                text_y = (y1_rel + y2_rel) / 2 - 15
                canvas.create_text(
                    text_x, text_y,
                    text="Connected",
                    fill="#FFFFFF",
                    font=("Segoe UI", 12, "bold")
                )
                
                # Create a pulsing effect for the connection line
                def pulse_line(count=0):
                    # Maximum of 6 pulses (3 seconds)
                    if count >= 6:
                        # Fade out and destroy
                        fade_out()
                        return
                    
                    # Toggle dash pattern for pulsing effect
                    dash_offset = count * 7
                    canvas.itemconfig(connection_line, dashoffset=dash_offset)
                    
                    # Schedule next pulse
                    animation_window.after(500, lambda: pulse_line(count + 1))
                
                # Fade out animation
                def fade_out(alpha=100):
                    if alpha <= 0:
                        animation_window.destroy()
                        return
                    
                    # Reduce alpha
                    animation_window.attributes("-alpha", alpha / 100)
                    animation_window.after(50, lambda: fade_out(alpha - 5))
                
                # Start pulsing animation
                pulse_line()
                return
                
            # Clear canvas
            canvas.delete("all")
            
            # Calculate progress (0 to 1)
            progress = min(1.0, step / max_steps)
            
            # Determine how many points to show based on progress
            points_to_show = int(len(curve_points) * progress)
            
            if points_to_show > 1:
                # Prepare points for the line
                points_flat = []
                for i in range(points_to_show):
                    x, y = curve_points[i]
                    points_flat.extend([x, y])
                
                # Draw partial connection line
                canvas.create_line(
                    points_flat,
                    width=4, 
                    fill=UIComponents.ANIMATION_COLOR_ACTIVE,
                    smooth=True
                )
            
            # Draw animated circle at the end of the visible line
            if points_to_show > 0:
                x, y = curve_points[points_to_show - 1]
                # Pulsing circle size
                size = 6 + 3 * math.sin(step * 0.8)
                canvas.create_oval(
                    x - size, y - size,
                    x + size, y + size,
                    fill=UIComponents.ANIMATION_COLOR_ACTIVE,
                    outline="#FFFFFF",
                    width=2
                )
                
                # Add trailing effect
                for i in range(3):
                    trail_idx = max(0, points_to_show - 3 + i)
                    if trail_idx < points_to_show:
                        tx, ty = curve_points[trail_idx]
                        trail_size = (i + 1) * 2
                        canvas.create_oval(
                            tx - trail_size, ty - trail_size,
                            tx + trail_size, ty + trail_size,
                            fill=UIComponents.ANIMATION_COLOR_ACTIVE,
                            stipple="gray50"  # Make it semi-transparent
                        )
            
            # Schedule next frame
            animation_window.after(50, lambda: animate_connection(step + 1, max_steps))
        
        # Start animation
        animate_connection()
    
    @staticmethod
    def end_call(app):
        """End the current active call."""
        if not app.call_active:
            return
            
        # Prevent multiple end call operations
        if getattr(app, 'is_ending_call', False):
            return
            
        app.is_ending_call = True
        
        # Add ending animation with countdown
        if hasattr(app, 'status_label'):
            app.status_label.config(text="Ending call in 3...")
            
            def countdown(count):
                if count <= 0:
                    # Countdown complete, actually end the call
                    CallHandlers._complete_end_call(app)
                    return
                    
                # Update countdown display
                app.status_label.config(text=f"Ending call in {count}...")
                
                # Flash the call button
                if hasattr(app, 'call_btn'):
                    current_color = app.call_btn.itemcget("button_bg", "fill")
                    
                    # Flash between red and gray
                    if current_color == "#E53935":  # Red
                        app.call_btn.itemconfig("button_bg", fill="#9E9E9E")  # Gray
                    else:
                        app.call_btn.itemconfig("button_bg", fill="#E53935")  # Red
                        
                # Continue countdown
                app.root.after(1000, lambda: countdown(count - 1))
                
            # Start countdown
            countdown(3)
        else:
            # If no status label, just end the call immediately
            CallHandlers._complete_end_call(app)
    
    @staticmethod
    def _complete_end_call(app):
        """Complete the call ending process after animations finish."""
        # Find the other app instance
        other_app = None
        for instance in CallHandlers.instances:
            if instance != app and instance.call_active:
                other_app = instance
                break
                
        # End call for both apps
        if other_app:
            # End call for the other app first
            if not getattr(other_app, 'is_ending_call', False):
                other_app.is_ending_call = True
                CallHandlers._terminate_call(other_app)
                
        # Now end call for this app
        CallHandlers._terminate_call(app)
        
        # Show call ended toast notification
        CallHandlers._show_call_ended_toast(app)
        
        # Save conversation to text file
        CallHandlers.save_conversation_to_text_file()
        
        # Clear the active call reference using app's class
        app.__class__.active_call = None
    
    @staticmethod
    def _show_call_ended_toast(app):
        """Show a toast notification that the call has ended."""
        if not hasattr(app, 'main_frame'):
            return
            
        # Create toast frame
        toast_frame = tk.Frame(
            app.main_frame,
            bg="#323232",  # Dark gray background
            padx=15,
            pady=10
        )
        
        # Position it at the bottom center
        toast_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        
        # Add message with icon
        toast_label = tk.Label(
            toast_frame,
            text=f"{MessageHandlers.CALL_EMOJI} Call Ended",
            font=("Segoe UI", 12),
            fg="white",
            bg="#323232"
        )
        toast_label.pack()
        
        # Add duration if available
        if app.call_start_time:
            duration = int(time.time() - app.call_start_time)
            minutes = duration // 60
            seconds = duration % 60
            
            duration_label = tk.Label(
                toast_frame,
                text=f"Duration: {minutes:02d}:{seconds:02d}",
                font=("Segoe UI", 10),
                fg="#BBBBBB",
                bg="#323232"
            )
            duration_label.pack()
            
        # Make toast disappear after a few seconds
        def remove_toast():
            toast_frame.destroy()
            
        app.root.after(3000, remove_toast)
        
        # Add fade out animation
        def fade_out(alpha=100):
            if alpha <= 0:
                remove_toast()
                return
                
            # Calculate transparency for Windows (different from other platforms)
            if sys.platform == "win32":
                # On Windows, use attributes to set transparency level
                try:
                    toast_frame.attributes("-alpha", alpha/100)
                except:
                    pass
            else:
                # On other platforms, simulate fading by changing background color
                gray_level = int(50 + (alpha/100) * 50)
                bg_color = f"#{gray_level:02x}{gray_level:02x}{gray_level:02x}"
                toast_frame.config(bg=bg_color)
                toast_label.config(bg=bg_color)
                if 'duration_label' in locals():
                    duration_label.config(bg=bg_color)
                    
            # Continue fading
            app.root.after(50, lambda: fade_out(alpha - 5))
            
        # Start fade out after 2 seconds
        app.root.after(2000, lambda: fade_out())
        
    @staticmethod
    def _terminate_call(app):
        """Terminate the call and reset UI state."""
        # Reset call state
        app.call_active = False
        app.is_ending_call = False
        
        # Reset connected_to reference
        app.connected_to = None
        
        # Update status
        if hasattr(app, 'status_label'):
            app.status_label.config(text="Call ended")
            
        # Reset call button to green "start call" button
        if hasattr(app, 'call_btn'):
            app.call_btn.itemconfig("button_bg", fill="#00BFA5")  # Green for start call
            app.call_btn.itemconfig("button_icon", text="📞")  # Reset icon
            
        # Remove call timer
        if app.duration_timer:
            app.root.after_cancel(app.duration_timer)
            app.duration_timer = None
            
        # Remove active call glow effect
        if hasattr(app, 'avatar_canvas'):
            app.avatar_canvas.delete("active_call_glow")
            
        # Reset name label color
        if hasattr(app, 'name_label'):
            app.name_label.config(fg="white")  # Reset to default color
            
    @staticmethod
    def update_duration(app):
        """Update the call duration display."""
        if not app.call_active:
            return
            
        # Calculate elapsed time
        elapsed = int(time.time() - app.call_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        
        # Update duration display
        if hasattr(app, 'duration_label'):
            app.duration_label.config(text=f"{minutes:02d}:{seconds:02d}")
        elif hasattr(app, 'status_label'):
            current_text = app.status_label.cget("text")
            # Only update if the text doesn't have a dynamic indicator
            if not "..." in current_text:
                app.status_label.config(text=f"Call active ({minutes:02d}:{seconds:02d})")
                
        # Periodically change text color to add visual interest 
        if elapsed % 5 == 0 and hasattr(app, 'duration_label'):
            original_color = app.duration_label.cget("fg")
            app.duration_label.config(fg="#4CAF50")  # Highlight color
            
            # Reset color after a brief flash
            def revert_color():
                if hasattr(app, 'duration_label') and app.duration_label.winfo_exists():
                    app.duration_label.config(fg=original_color)
                    
            app.root.after(500, revert_color)
            
        # Schedule next update
        app.duration_timer = app.root.after(1000, lambda: CallHandlers.update_duration(app))
        
    @staticmethod
    def save_conversation_to_text_file():
        """Save the conversation history to a text file."""
        # Create directory for conversation logs if it doesn't exist
        os.makedirs("conversation_history", exist_ok=True)
        
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history/call_{timestamp}.txt"
        
        # Format the conversation with proper timestamps and icons
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== CONVERSATION HISTORY ({timestamp}) ===\n\n")
            
            # Worker conversation
            f.write(f"=== WORKER CONVERSATION ===\n")
            for msg in CallHandlers.worker_conversation:
                f.write(f"{msg}\n")
                
            f.write("\n")
            
            # Supervisor conversation
            f.write(f"=== SUPERVISOR CONVERSATION ===\n")
            for msg in CallHandlers.supervisor_conversation:
                f.write(f"{msg}\n")
                
        # Print to terminal that the conversation was saved
        print(f"Conversation saved to {filename}")
        
        # Return the filename for reference
        return filename 