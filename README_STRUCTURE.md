# Voice Call App - Code Structure

## Overview

This project has been refactored to follow better software engineering practices, with the codebase split into multiple files for improved organization, maintainability, and readability. The original monolithic `template.py` has been divided into the following files:

## File Structure

- **app.py**: Main entry point for the application
- **voice_call_app.py**: Core VoiceCallApp class that delegates to the specialized handler classes
- **redirector.py**: Contains the StdoutRedirector class for capturing and directing output
- **ui_components.py**: Contains UI setup functions separated from core logic
- **message_handlers.py**: Handles message processing and display functionality
- **call_handlers.py**: Manages call-related functionality

## Component Responsibilities

### app.py
- Creates the root windows for the Worker and Supervisor apps
- Initializes the VoiceCallApp instances
- Positions windows and starts the application

### voice_call_app.py
- Main class that delegates functionality to specialized handler classes
- Maintains application state
- Exposes methods that delegate to the appropriate handlers

### redirector.py
- Captures stdout output using StringIO redirection
- Processes captured output to update UI in real-time
- Handles message formatting based on source (Worker/Supervisor)

### ui_components.py
- Contains methods for setting up the UI elements
- Creates text areas, buttons, labels, and other UI components
- Configures UI styles and layout

### message_handlers.py
- Manages message display, formatting, and animation
- Handles the 5-second delay for message transitions
- Implements thinking indicators and conversation display

### call_handlers.py
- Manages call state, connections between instances
- Handles call setup, termination, and duration tracking
- Implements the threading logic for non-blocking operation

## Improvements Made

1. **Separation of Concerns**: Each file now focuses on a specific aspect of the application
2. **Reduced File Size**: No single file is overly large or complex
3. **Better Organization**: Related functions are grouped together
4. **Maintainability**: Easier to find and fix issues in specific areas
5. **Code Reuse**: Common functionality is centralized
6. **Delegation Pattern**: Main class delegates to specialized handlers

## Running the Application

```bash
python app.py
```

This will start both the Worker and Supervisor interfaces. Enter a message in either interface and press "Start Calling" to begin a conversation.

## Technical Requirements

- Python 3.6+ 
- Tkinter (usually included with Python)
- `main.py` from the original project (used for AI processing) 