import tkinter as tk
import time
from datetime import datetime
import threading
import re
import json
import os

class MessageHandlers:
    # Track whether to preserve conversation history
    preserve_history = True  # Changed to True by default
    
    # Class variables to store thinking state icons
    worker_thinking_icon = "★ Worker (Local) is thinking... 💭"
    supervisor_thinking_icon = "★ Supervisor (Remote) is thinking... 💭"
    
    # Enhanced emoji set for rich visual feedback
    WORKER_EMOJI = "👨‍💻"
    SUPERVISOR_EMOJI = "👩‍💼"
    SYSTEM_EMOJI = "🔄"
    CALL_EMOJI = "📞"
    SAVE_EMOJI = "💾"
    WARNING_EMOJI = "⚠️"
    SUCCESS_EMOJI = "✅"
    ERROR_EMOJI = "❌"
    THINKING_EMOJI = "💭"
    QUESTION_EMOJI = "❓"
    SPEAKING_EMOJI = "💬"
    CODE_EMOJI = "📝"
    SEND_EMOJI = "📤"
    RECEIVE_EMOJI = "📥"
    
    # Message action icons
    COPY_ICON = "📋"
    REPLY_ICON = "↩️"
    DELETE_ICON = "🗑️"
    EDIT_ICON = "✏️"
    FORWARD_ICON = "➡️"
    
    # Additional UI enhancement icons
    ATTACHMENT_ICON = "📎"
    CALENDAR_ICON = "📅"
    CLOCK_ICON = "⏰"
    LOCATION_ICON = "📍"
    CAMERA_ICON = "📷"
    MIC_ICON = "🎤"
    SEARCH_ICON = "🔍"
    SETTINGS_ICON = "⚙️"
    INFO_ICON = "ℹ️"
    LINK_ICON = "🔗"
    LIKE_ICON = "👍"
    DISLIKE_ICON = "👎"
    STAR_ICON = "⭐"
    PIN_ICON = "📌"
    NEW_MESSAGE_ICON = "🔔"
    
    # Status indicators
    ONLINE_STATUS = "🟢"
    AWAY_STATUS = "🟠"
    BUSY_STATUS = "🔴"
    OFFLINE_STATUS = "⚫"
    
    # For saving conversation history
    conversation_history_dir = "conversation_history"
    
    @staticmethod
    def toggle_preserve_history(app, preserve=None):
        """Toggle or set whether to preserve conversation history.
        
        Args:
            app: The application instance
            preserve: If provided, sets the preserve_history flag to this value.
                     If None, toggles the current value.
        
        Returns:
            The new preserve_history value
        """
        if preserve is None:
            MessageHandlers.preserve_history = not MessageHandlers.preserve_history
        else:
            MessageHandlers.preserve_history = preserve
            
        # Update UI to indicate history preservation status if needed
        if hasattr(app, 'status_label'):
            history_status = f"{MessageHandlers.SAVE_EMOJI} History: Preserved" if MessageHandlers.preserve_history else f"{MessageHandlers.SAVE_EMOJI} History: Not Preserved"
            if not app.call_active:
                app.status_label.config(text=history_status)
                
        return MessageHandlers.preserve_history
    
    @staticmethod
    def should_clear_text(app):
        """Check if text should be cleared based on history preservation setting.
        
        Returns:
            True if text should be cleared, False if history should be preserved
        """
        return not MessageHandlers.preserve_history
    
    @staticmethod
    def safe_clear_text(app):
        """Safely clear text respecting the history preservation setting."""
        if MessageHandlers.should_clear_text(app):
            app.response_text.delete("1.0", tk.END)
    
    @staticmethod
    def clear_response(app):
        """Clear the response text area."""
        app.response_text.config(state=tk.NORMAL)
        MessageHandlers.safe_clear_text(app)
        app.response_text.config(state=tk.DISABLED)
    
    @staticmethod
    def _ensure_autoscroll(app):
        """Ensure text widget is scrolled to the end and updates are visible."""
        try:
            # Force scrolling with a more aggressive approach
            app.response_text.see(tk.END)
            app.response_text.update_idletasks()
            
            # Calculate the last visible line
            last_visible_index = app.response_text.index("@0,%d" % app.response_text.winfo_height())
            last_line_index = app.response_text.index(tk.END + "-1c")
            
            # Check if we need to force scroll
            if last_visible_index != last_line_index:
                # More aggressive scrolling if needed
                app.response_text.yview_moveto(1.0)  # Force scroll to 100%
                app.response_text.update_idletasks()
            
            # Schedule multiple delayed scrolls for reliability
            def force_scroll_now():
                try:
                    if app.response_text.winfo_exists():
                        app.response_text.see(tk.END)
                        app.response_text.yview_moveto(1.0)
                        app.response_text.update_idletasks()
                except Exception:
                    pass
                    
            def force_scroll_later():
                try:
                    if app.response_text.winfo_exists():
                        app.response_text.see(tk.END)
                        app.response_text.yview_moveto(1.0)
                        # Extra step to ensure the view is at the bottom
                        app.response_text.mark_set("insert", tk.END)
                        app.response_text.see("insert")
                        app.response_text.update_idletasks()
                except Exception:
                    pass
                    
            def force_scroll_final():
                try:
                    if app.response_text.winfo_exists():
                        # Most aggressive approach - jump directly to the end
                        app.response_text.yview_moveto(1.0)
                        last_line = app.response_text.index(tk.END + "-1c linestart")
                        app.response_text.mark_set("insert", last_line)
                        app.response_text.see("insert")
                        app.response_text.update_idletasks()
                except Exception:
                    pass
                
            # Schedule the scrolls at different times to ensure at least one works
            app.root.after(50, force_scroll_now)
            app.root.after(150, force_scroll_later)
            app.root.after(300, force_scroll_final)
            
            # One final scroll after all content should be rendered
            app.root.after(500, lambda: app.response_text.yview_moveto(1.0))
            
        except Exception as e:
            print(f"Error in auto-scroll: {e}")

    @staticmethod
    def _remove_thinking_indicators(app):
        """Remove any thinking indicators from the text."""
        try:
            if not app.response_text.winfo_exists():
                return
                
            app.response_text.config(state=tk.NORMAL)
            
            # Get current content
            current_text = app.response_text.get("1.0", tk.END)
            
            # Check for and remove any thinking indicators
            for indicator in [MessageHandlers.worker_thinking_icon, MessageHandlers.supervisor_thinking_icon]:
                if indicator in current_text:
                    # Find all occurrences and remove them
                    start_idx = 1.0
                    while True:
                        # Search for the indicator from the current start index
                        start_idx = app.response_text.search(indicator, start_idx, tk.END)
                        if not start_idx:
                            break
                        
                        # Find the end of the line containing the indicator
                        line_start = app.response_text.index(f"{start_idx} linestart")
                        line_end = app.response_text.index(f"{start_idx} lineend+1c")
                        
                        # Delete the entire line containing the indicator
                        app.response_text.delete(line_start, line_end)
            
            app.response_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error removing thinking indicators: {e}")
            
    @staticmethod
    def _preserve_thinking_indicator(app, indicator, tag):
        """Insert a thinking indicator that will persist even across updates."""
        try:
            # Check if indicator already exists
            current_text = app.response_text.get("1.0", tk.END)
            if indicator in current_text:
                return  # Indicator already exists, no need to add it
                
            # Add the thinking indicator at the end
            app.response_text.config(state=tk.NORMAL)
            
            # Add a separator if there's content and we're preserving history
            if MessageHandlers.preserve_history and current_text.strip():
                app.response_text.insert(tk.END, "\n\n", "")
                
            app.response_text.insert(tk.END, indicator + "\n", tag)
            
            # Make sure it's visible
            MessageHandlers._ensure_autoscroll(app)
            app.response_text.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error preserving thinking indicator: {e}")
    
    @staticmethod
    def _format_message_with_actions(app, message, sender):
        """Format message with interactive action buttons"""
        # Add action buttons after messages
        if hasattr(app, 'response_text'):
            # Get current position for inserting action buttons
            current_pos = app.response_text.index(tk.END)
            
            # Create interactive action bar with buttons
            action_bar = f" {MessageHandlers.COPY_ICON} {MessageHandlers.REPLY_ICON} {MessageHandlers.LIKE_ICON}"
            
            # For code blocks, add a special copy code button
            if "```" in message:
                action_bar += f" {MessageHandlers.CODE_EMOJI}"
                
            return message + "\n" + action_bar + "\n"
        
        return message + "\n"
    
    @staticmethod
    def _add_timestamp_to_message(message):
        """Add timestamp to message for better context"""
        timestamp = datetime.now().strftime("%H:%M")
        return f"{message} {MessageHandlers.CLOCK_ICON} {timestamp}"
    
    @staticmethod
    def append_to_response(app, text):
        """Append text to the response area with enhanced formatting."""
        if not hasattr(app, 'response_text'):
            return
            
        # Determine sender based on the app's model label
        is_worker = app.model_label == "Worker (Local)"
        sender_emoji = MessageHandlers.WORKER_EMOJI if is_worker else MessageHandlers.SUPERVISOR_EMOJI
        sender_label = "Worker" if is_worker else "Supervisor"
        
        # Enhanced formatting with sender info and timestamp
        formatted_text = f"{sender_emoji} {sender_label}: {text}"
        formatted_text = MessageHandlers._add_timestamp_to_message(formatted_text)
        formatted_text = MessageHandlers._format_message_with_actions(app, formatted_text, sender_label)
        
        # Apply formatting and colorization
        tag = "worker_message" if is_worker else "supervisor_message"
        MessageHandlers._update_response_text(app, formatted_text, tag)
        
        # Ensure the conversation is visible with auto-scroll
        MessageHandlers._ensure_autoscroll(app)
        
        # Store the message in conversation history if enabled
        if MessageHandlers.preserve_history:
            # Save message to history
            pass
            
        # Force exit thinking state if it was active
        MessageHandlers._force_exit_thinking_state(app)

    @staticmethod
    def show_thinking_in_response(app, is_thinking=True):
        """Shows or hides the thinking indicator in the response area."""
        if is_thinking:
            app.is_thinking = True
            
            # Schedule thinking animation updates
            app.root.after(100, lambda: MessageHandlers._delayed_thinking_update(app))
            
            # Force exit thinking state after a timeout (2 minutes)
            app.root.after(120000, lambda: MessageHandlers._force_exit_thinking_state_if_still_thinking(app))
                
            # Only show thinking status relevant to this app instance
            if app.model_label == "Supervisor (Remote)":
                thinking_message = MessageHandlers.supervisor_thinking_icon
                # Update the status label
                app.status_label.config(text=thinking_message)
                
                # Add the thinking indicator in a way that persists
                MessageHandlers._preserve_thinking_indicator(app, thinking_message, "supervisor_thinking")
                
            elif app.model_label == "Worker (Local)":
                thinking_message = MessageHandlers.worker_thinking_icon
                # Update the status label
                app.status_label.config(text=thinking_message)
                
                # Add the thinking indicator in a way that persists
                MessageHandlers._preserve_thinking_indicator(app, thinking_message, "worker_thinking")
                
            else:
                # Generic handling for any other model labels
                thinking_message = f"★ {app.model_label} is thinking... ★"
                app.status_label.config(text=thinking_message)
            
            # Use the force exit method for consistency
            MessageHandlers._force_exit_thinking_state(app)
            
    @staticmethod
    def _delayed_thinking_update(app):
        """Handle delayed update for thinking state."""
        try:
            if not app.response_text.winfo_exists() or not app.is_thinking:
                return
                
            # Define the animated thinking indicator (with subtle animation)
            import random
            dots = ["...", ". ..", ".. .", "..."]
            dot_pattern = dots[random.randint(0, 3)]
            
            # Update the thinking indicator based on model
            if app.model_label == "Worker (Local)":
                # Update status label with animation
                app.status_label.config(text=f"Worker is thinking{dot_pattern}")
                
                # Make sure thinking indicator is still visible in text
                MessageHandlers._preserve_thinking_indicator(app, MessageHandlers.worker_thinking_icon, "worker_thinking")
                
            elif app.model_label == "Supervisor (Remote)":
                # Update status label with animation
                app.status_label.config(text=f"Supervisor is thinking{dot_pattern}")
                
                # Make sure thinking indicator is still visible in text
                MessageHandlers._preserve_thinking_indicator(app, MessageHandlers.supervisor_thinking_icon, "supervisor_thinking")
            
            # Schedule next update with a varying delay for more natural animation
            delay = random.randint(400, 800)
            if app.is_thinking:
                app.root.after(delay, lambda: MessageHandlers._delayed_thinking_update(app))
                
        except Exception as e:
            print(f"Error in thinking update: {e}")

    @staticmethod
    def _force_exit_thinking_state_if_still_thinking(app):
        """Force exit thinking state only if still in thinking state."""
        if app.is_thinking:
            MessageHandlers._force_exit_thinking_state(app)

    @staticmethod
    def _force_exit_thinking_state(app):
        """Force exit thinking state."""
        # Reset thinking flag
        app.is_thinking = False
            
        # Clear any thinking-related text
        app.response_text.config(state=tk.NORMAL)
        
        # Remove all thinking indicators
        MessageHandlers._remove_thinking_indicators(app)
        
        app.response_text.config(state=tk.DISABLED)
        
        # Reset status label
        if app.status_label.winfo_exists():
            app.status_label.config(text="")
            
    @staticmethod
    def _update_response_text(app, text, tag):
        """Update the response text with new content."""
        try:
            if not app.response_text.winfo_exists():
                return
                
            # Enable text widget for editing
            app.response_text.config(state=tk.NORMAL)
            
            # First remove any existing thinking indicators
            MessageHandlers._remove_thinking_indicators(app)
            
            # Clear any existing content if history preservation is off
            MessageHandlers.safe_clear_text(app)
            
            # Get current timestamp for the message
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Improve the quality of the text
            text = MessageHandlers._improve_message_quality(text)
            
            # Determine who's sending the message based on tag and app type
            # Only show Worker messages in Worker app and Supervisor messages in Supervisor app
            if tag == "worker_message":
                is_question = ("?" in text and not "Waiting for" in text)
                header_text = f"[{current_time}] {MessageHandlers.WORKER_EMOJI} Worker"
                header_text += " asks:" if is_question else " says:"
                
                # Only display Worker messages in the Worker app
                if app.model_label == "Worker (Local)":
                    app.response_text.insert(tk.END, header_text + "\n", "worker_header")
                    app.response_text.insert(tk.END, text, "worker_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                # In Supervisor app, only show "Worker says:" messages, not the full worker conversation
                elif app.model_label == "Supervisor (Remote)" and text.startswith("Worker:"):
                    app.response_text.insert(tk.END, header_text + "\n", "worker_header")
                    app.response_text.insert(tk.END, text.replace("Worker: ", ""), "worker_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                
            elif tag == "supervisor_message":
                is_question = ("?" in text and not "Waiting for" in text)
                header_text = f"[{current_time}] {MessageHandlers.SUPERVISOR_EMOJI} Supervisor"
                header_text += " asks:" if is_question else " says:"
                
                # Only display Supervisor messages in the Supervisor app
                if app.model_label == "Supervisor (Remote)":
                    app.response_text.insert(tk.END, header_text + "\n", "supervisor_header")
                    app.response_text.insert(tk.END, text, "supervisor_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                # In Worker app, only show "Supervisor says:" messages, not the full supervisor conversation
                elif app.model_label == "Worker (Local)" and text.startswith("Supervisor:"):
                    app.response_text.insert(tk.END, header_text + "\n", "supervisor_header")
                    app.response_text.insert(tk.END, text.replace("Supervisor: ", ""), "supervisor_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                
            else:
                # For system messages or other types - show in both apps
                app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.SYSTEM_EMOJI} System: ", "system_header")
                app.response_text.insert(tk.END, text + "\n", tag)
                
            # Add a subtle separator if preserving history
            if MessageHandlers.preserve_history:
                app.response_text.insert(tk.END, "─" * 50 + "\n", "light_separator")
            
            # Make sure the text is visible with our enhanced auto-scroll
            MessageHandlers._ensure_autoscroll(app)
            
            # Disable text widget to prevent editing
            app.response_text.config(state=tk.DISABLED)
            
            # Update UI
            app.root.update_idletasks()
        except Exception as e:
            print(f"Error updating response text: {e}")
            
    @staticmethod
    def _format_code_block(text):
        """Format code blocks properly with syntax highlighting."""
        if "```" not in text:
            return text
            
        # Process code blocks specially
        result = []
        parts = text.split("```")
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                result.append(part)
            else:  # Code block
                # Check for language specifier
                lines = part.split("\n", 1)
                language = "plain"
                code = part
                
                if len(lines) > 1 and lines[0].strip():
                    # Has language specifier
                    language = lines[0].strip()
                    code = lines[1]
                    
                # Format with special tag marking for later application
                result.append(f"\n<<<CODE_BLOCK_START:{language}>>>\n{code}\n<<<CODE_BLOCK_END>>>\n")
                
        return "".join(result)
        
    @staticmethod
    def _apply_code_block_formatting(app, text):
        """Apply code block formatting to text with code block markers."""
        if "<<<CODE_BLOCK_START:" not in text:
            return False
            
        # Find all code blocks and apply formatting
        while "<<<CODE_BLOCK_START:" in text:
            # Get positions
            start_marker = "<<<CODE_BLOCK_START:"
            end_marker = "<<<CODE_BLOCK_END>>>"
            
            start_pos = text.find(start_marker)
            lang_end_pos = text.find(">>>", start_pos)
            end_pos = text.find(end_marker, lang_end_pos)
            
            if start_pos == -1 or lang_end_pos == -1 or end_pos == -1:
                break
                
            # Extract parts
            language = text[start_pos + len(start_marker):lang_end_pos]
            code_block = text[lang_end_pos + 3:end_pos]
            
            # Remove the markers from text
            text = text[:start_pos] + text[end_pos + len(end_marker):]
            
            # Insert formatted code block
            current_insert = app.response_text.index(tk.INSERT)
            app.response_text.insert(current_insert, f"[Code block - {language}]\n", "code_header")
            app.response_text.insert(tk.INSERT, code_block + "\n", "code_block")
            
        return True
            
    @staticmethod
    def _improve_message_quality(text):
        """Improve the quality of message display."""
        # If text is empty, return as is
        if not text:
            return text
        
        # Handle code blocks specially with proper formatting
        text = MessageHandlers._format_code_block(text)
        
        # If the text contains code block markers, special handling is needed
        if "<<<CODE_BLOCK_START:" in text:
            return text
            
        # Remove excessive whitespace but preserve paragraph breaks
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []
        
        for paragraph in paragraphs:
            # Join lines within paragraphs and clean up whitespace
            paragraph = ' '.join(line.strip() for line in paragraph.split("\n"))
            paragraph = ' '.join(paragraph.split())
            cleaned_paragraphs.append(paragraph)
            
        text = "\n\n".join(cleaned_paragraphs)
            
        # Clean up common issues
        text = text.replace(" , ", ", ")
        text = text.replace(" . ", ". ")
        text = text.replace(" : ", ": ")
        text = text.replace(" ; ", "; ")
        
        # Ensure proper spacing after punctuation
        for punctuation in ['.', '!', '?']:
            text = text.replace(f"{punctuation}", f"{punctuation} ")
            text = text.replace(f"{punctuation}  ", f"{punctuation} ")
        
        # Fix multiple spaces
        while "  " in text:
            text = text.replace("  ", " ")
        
        # Break very long paragraphs for better readability
        if len(text) > 400 and "\n\n" not in text:
            words = text.split()
            chunks = []
            current_chunk = []
            
            for word in words:
                current_chunk.append(word)
                if len(' '.join(current_chunk)) > 80 and word.endswith(('.', '!', '?')):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            text = '\n\n'.join(chunks)
        
        # Handle bullet points and numbered lists better
        lines = text.split('\n')
        for i in range(len(lines)):
            # Check for bullet points or numbered lists
            if (lines[i].strip().startswith('- ') or 
                lines[i].strip().startswith('* ') or 
                (lines[i].strip() and lines[i].strip()[0].isdigit() and lines[i].strip()[1:].startswith('. '))):
                # For bullet points or lists, add extra spacing before the item
                if i > 0 and not lines[i-1].strip().endswith(':'): 
                    lines[i] = '\n' + lines[i]
        
        text = '\n'.join(lines)
            
        return text.strip()
            
    @staticmethod
    def _delayed_text_update(app, text, tag):
        """Handle delayed update of text."""
        try:
            if not app.response_text.winfo_exists():
                return
                
            # Get current timestamp for the message
            current_time = datetime.now().strftime("%H:%M:%S")
                
            # Store previous conversation content if it exists
            app.response_text.config(state=tk.NORMAL)
            
            # First remove any existing thinking indicators
            MessageHandlers._remove_thinking_indicators(app)
            
            previous_content = app.response_text.get("1.0", tk.END).strip()
            
            # Format the message for better quality
            text = MessageHandlers._improve_message_quality(text)
            
            # Clear any thinking indicators or empty messages
            if "is thinking" in previous_content or "ready for new input" in previous_content or not previous_content:
                MessageHandlers.safe_clear_text(app)
                
                # Format and insert the new content with timestamp and emoji
                if tag == "worker_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    header_text = f"[{current_time}] {MessageHandlers.WORKER_EMOJI} Worker"
                    header_text += " asks:" if is_question else " says:"
                    app.response_text.insert(tk.END, header_text + "\n", "worker_header")
                    app.response_text.insert(tk.END, text, "worker_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                elif tag == "supervisor_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    header_text = f"[{current_time}] {MessageHandlers.SUPERVISOR_EMOJI} Supervisor"
                    header_text += " asks:" if is_question else " says:"
                    app.response_text.insert(tk.END, header_text + "\n", "supervisor_header")
                    app.response_text.insert(tk.END, text, "supervisor_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                else:
                    # For system messages or other types
                    app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.SYSTEM_EMOJI} System: ", "system_header")
                    app.response_text.insert(tk.END, text + "\n", tag)
            else:
                # If preserving history and there's existing content,
                # add a more prominent separator to indicate new conversation segment
                if MessageHandlers.preserve_history:
                    app.response_text.insert(tk.END, "\n\n" + "═" * 60 + "\n\n", "separator")
                else:
                    # If not preserving history, clear before adding new content
                    MessageHandlers.safe_clear_text(app)
                
                # Format and insert the new content with timestamp and emoji
                if tag == "worker_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    header_text = f"[{current_time}] {MessageHandlers.WORKER_EMOJI} Worker"
                    header_text += " asks:" if is_question else " says:"
                    app.response_text.insert(tk.END, header_text + "\n", "worker_header")
                    app.response_text.insert(tk.END, text, "worker_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                elif tag == "supervisor_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    header_text = f"[{current_time}] {MessageHandlers.SUPERVISOR_EMOJI} Supervisor"
                    header_text += " asks:" if is_question else " says:"
                    app.response_text.insert(tk.END, header_text + "\n", "supervisor_header")
                    app.response_text.insert(tk.END, text, "supervisor_message")
                    app.response_text.insert(tk.END, "\n", "")  # Add newline after message
                else:
                    # For system messages or other types
                    app.response_text.insert(tk.END, f"[{current_time}] {MessageHandlers.SYSTEM_EMOJI} System: ", "system_header")
                    app.response_text.insert(tk.END, text + "\n", tag)
            
            # Add a subtle separator after the message if preserving history
            if MessageHandlers.preserve_history:
                app.response_text.insert(tk.END, "─" * 50 + "\n", "light_separator")
            
            # Make sure the new text is visible with enhanced auto-scroll
            MessageHandlers._ensure_autoscroll(app)
            
            # Reset to read-only
            app.response_text.config(state=tk.DISABLED)
            
            # Update the status label based on who's speaking
            if app.status_label.winfo_exists():
                if tag == "worker_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    status_text = "Worker asking" if is_question else "Worker speaking"
                    app.status_label.config(text=status_text)
                elif tag == "supervisor_message":
                    is_question = ("?" in text and not "Waiting for" in text)
                    status_text = "Supervisor asking" if is_question else "Supervisor speaking"
                    app.status_label.config(text=status_text)
                else:
                    app.status_label.config(text="Conversation active")
        except Exception as e:
            print(f"Error in delayed text update: {e}") 

    @staticmethod
    def save_conversation_history(app):
        """Save the conversation history to a JSON file."""
        # Ensure the directory exists
        try:
            os.makedirs(MessageHandlers.conversation_history_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory: {e}")
            if hasattr(app, 'status_label'):
                app.status_label.config(text=f"{MessageHandlers.ERROR_EMOJI} Error creating directory")
            return None
            
        # Get conversation text
        if not hasattr(app, 'response_text') or not app.response_text.winfo_exists():
            return None
            
        conversation_text = app.response_text.get("1.0", tk.END)
        if not conversation_text.strip():
            return None
            
        # Prepare data structure
        conversation_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "app_type": "worker" if app.model_label == "Worker (Local)" else "supervisor",
            "model_name": app.model_name,
            "conversation_text": conversation_text,
            "call_duration": app.call_duration if hasattr(app, 'call_duration') else None
        }
        
        # Generate a filename with timestamp and app type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        app_type = "worker" if app.model_label == "Worker (Local)" else "supervisor"
        filename = f"{MessageHandlers.conversation_history_dir}/conversation_{app_type}_{timestamp}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            # Provide feedback on successful save
            if hasattr(app, 'status_label'):
                app.status_label.config(text=f"{MessageHandlers.SAVE_EMOJI} Conversation saved to {os.path.basename(filename)}")
                
            return filename
        except Exception as e:
            print(f"Error saving conversation history: {e}")
            if hasattr(app, 'status_label'):
                app.status_label.config(text=f"{MessageHandlers.ERROR_EMOJI} Error saving conversation: {str(e)}")
            return None 