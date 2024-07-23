from PCBuddyNew import State
import subprocess

def execute_function_node(state : State):
    code = state['messages'][-1]["content"]
    try:
        # Use subprocess.run to execute the code with capture output
        print("Executing....")
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, check=True
        )
        print("Executed..")
        print("Result :", result)
        print("Executed and after result..")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Handle errors during execution
        return f"Error: {e}"
    except Exception as e:
    # Handle any other unexpected errors
        return f"Unexpected Error: {str(e)}"

