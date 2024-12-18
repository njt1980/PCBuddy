import subprocess,os

def execute_code(code):
    """
    This function executes the provided Python code snippet

    Args:
        code (str): The Python code snippet to be executed.

    Returns:
        str: The standard output of the executed code,
             or an error message if execution fails.
    """
    print("Code :", code)
    script_directory = os.path.dirname(os.path.abspath(__file__))
    print(f"Changing directory to: {script_directory}")
    os.chdir(script_directory)
    print(f"Current directory after change: {os.getcwd()}")
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

def llm_tool_execute_code(llm_generated_code):
    """
    Executes the Python code generated by the LLM and returns the result.

    Parameters:
    llm_generated_code (str): The Python code generated by the LLM.

    Returns:
    dict: A dictionary containing 'status' (success or error) and 'result' (output or error message).
    """
    import contextlib
    import io

    print("Executing code:\n", llm_generated_code)

    # Capture the output and error
    output = io.StringIO()
    error = io.StringIO()

    try:
        print("Before exec..")
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            exec(llm_generated_code, globals(), locals())
        print("After exec..")
        result = output.getvalue().strip()
        print("Result :",result)
        if not result:
            result = "Code executed successfully with no output."
        print("Returning result..")
        
        return {'status': 'success', 'result': result}
    except Exception as e:
        error_message = f"Error: {str(e)}\n{error.getvalue().strip()}"
        print("Exception occurred:", error_message)
        return {'status': 'error', 'result': error_message}
    finally:
        output.close()
        error.close()
    
code = """
import os
print("Inside the executed code")
current_directory = os.getcwd()
print("Current Directory:", current_directory)
"""

print("First Function..")
execute_code(code)

print("Second Function..")
llm_tool_execute_code(code)

print("Done..")
