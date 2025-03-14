def execute_terminal_command(command: str):
    import subprocess
    
    try:
        # Execute the command and capture output
        result = subprocess.run(
            command, 
            shell=True, 
            text=True, 
            capture_output=True
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Error: {result.stderr}"
    except Exception as e:
        return False, f"Exception occurred: {str(e)}"