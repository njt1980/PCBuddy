import os
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Annotated
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import ensure_config
from langchain_core.tools import tool
from langgraph.graph import StateGraph,END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import subprocess
import psutil
# from HelperFunctions import execute_function_node
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model = "gpt-3.5-turbo-0125")
# llm = ChatOpenAI(model = "gpt-4o")

class State(TypedDict):
  messages : Annotated[list[AnyMessage], add_messages]  
  output : str
  next : str

agent_prompt = ChatPromptTemplate.from_messages(
    [
    ("system",
     "You are an expert code generating agent."
     "Your role is to generate the code which when executed will provide the user with the requested information"
     "Your response will be directly provided as input to a function which will execute it"
     "So do not include any non-executable statements in your response"
     "After the code which generates the user requested information or action,"
     "The code should always have a print statement which provides an answer/update to the user's query or request in spoken English"
     "using the response from the executed code"
     "For ex. User input : Which is my current folder?"
     "Response : import os\ncurrent_folder = os.getcwd()\nprint('Your current folder is ' + current_folder)"
     "Always ensure that the code is executable without any modification"
     "For example, do not include ```python at the start of the code as that would cause an error when executing"
     "\nCurrent Time: {time}"
    ),
    ("placeholder",
     "{messages}")
    ]
).partial(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class Assistant:
  def __init__(self,runnable : Runnable):
    self.runnable = runnable

  def __call__(self, state: State, config: RunnableConfig):
    while True:
      result = self.runnable.invoke(state)
      if not result.tool_calls and (
          not result.content
          or isinstance(result.content, list)
          and not result.content[0].get("text")
      ):
        messages = state["messages"] + [("user","Respond to user")]
        state = {**state, "messages": messages}
      else:
        break
    return {"messages": result}


assistant_runnable = agent_prompt | llm

def execute_function_node(state : State):
    print("State in execute function :", state)
    code = state['messages'][-1].content
    print("Code :", code)
    print("Code type :", type(code))
    try:
        # Use subprocess.run to execute the code with capture output
        print("Executing....")
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, check=True
        )
        print("Executed..")
        print("Result :", result)
        print("Executed and after result..")
        return {"output" : result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        # Handle errors during execution
        print("ERROR ERROR ERROR")
        print("Error is :", f"Error: {e}")
        error_message = f"Error: {e.stderr.strip()}"
        print("Error Message :", error_message)
        return {"output": f"Error: {e}"}
    except Exception as e:
    # Handle any other unexpected errors
        print("ERROR ERROR ERROR...")
        return {"output": f"Unexpected Error: {str(e)}"}



graph = StateGraph(State)

graph.add_node("agent", Assistant(assistant_runnable))

graph.add_node("executor_node",execute_function_node)

graph.add_edge("agent","executor_node")

graph.set_entry_point("agent")

memory = SqliteSaver.from_conn_string(":memory:")

graph = graph.compile(
    checkpointer=memory
)

thread_id = str(uuid.uuid4())

config = {"configurable":{
    "thread_id":thread_id,
}}

# events = graph.stream(
#     {"messages":("user","From which folder am I executing the code from?")},
#     config=config,
#     stream_mode="values"
# )

# for event in events:
#   print(event["messages"][-1].pretty_print()) 

# response = graph.invoke(
#     {"messages":("user","How many available space is there on my system?")},
#     config=config,
#     stream_mode="values"
# )

def execute_graph(usr_input : str):
   response = graph.invoke(
      {"messages":("user",usr_input)},
      config = config,
      stream_mode='values'
   )
   return response

# print("Response :", response)

print(os.getcwd())