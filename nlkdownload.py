import sys
import os

# Add the voice path to system path
sys.path.insert(0, os.path.abspath(r'C:\VC AI'))

# Change to the VC AI directory
os.chdir(r'C:\VC AI')

# Print the current directory
print(f"Current working directory: {os.getcwd()}")

# Run main.py
if __name__ == "__main__":
    try:
        # Import and execute main.py
        from main import *
        print("Successfully executed main.py")
    except Exception as e:
        print(f"Error executing main.py: {e}")
