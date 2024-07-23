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
from langchain_core.pydantic_v1 import BaseModel, Field
import ast, json
import subprocess
from typing import TypedDict
import gradio as gr
after_human :str 

class CodeResponse(BaseModel):
  code : str = Field(description = 'Code to be executed')
  summary : str = Field(description = 'Summary of the steps which will be taken to get the required information')
  risks : str = Field(description = 'Risks associated with executing the code')

class SearchResponse(BaseModel):
  response : str = Field(description = 'Response from LLM to user input')

class State(TypedDict):
  messages : Annotated[list[AnyMessage], add_messages]
  next_action : str
  coderesponse : str
  output : str
  after_human : str

class RouterResponse(BaseModel):
  """ Response from router. Needs to be either "Search" or "Code" """
  next_actor: str = Field(description = 'Response from router, i.e, "Search" or "Code"')

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful agent."
     "You will review the user input."
     "If the user query is related to their system/PC information for which you can generate python code, which can be executed to get the required"
     "information, response with 'Code', else respond with 'Search'"
     "If the input has multiple questions, respond with 'Search' "
    ),
    ("placeholder", "{messages}")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert python coder."
     "Based on the user request, you will respond with the corresponding code, summary of steps which will be taken and with any risks"
     "with executing the code."
     "Ensure that the response is a dict with the following keys, 'code', 'summary', 'risks'"
     "The 'code' key should only contain the code which needs to be executed"
     "The code should always have a print statement which provides an answer to the user's query or request in spoken English"
     "Include a function as below in the code"
     "def install(package):"
     " subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])"
     "Use a try-except loop for importing the required modules and to install the package with the above function, if the modules is not available."
     "It is ABSOLUTELY necessary that the code checks for the required modules and install them if not available"
     "if there is a 'import psutil' statement, then there should be a try-except as mentioned above,"
     "to check if psutil is available and to install it if not available"
     "The 'summary' key should contain the summary of the steps which will be taken to get the required information"
     "The 'risks' key should contain any risks associated with executing the code"

    ),
    ("placeholder", "{messages}")
])

search_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert support assistant."
     "You will review the user's query and respond accordingly"
     "If there are multiple questions not related to each other, "
     "respond asking the user to stick to one topic."
    ),
    ("placeholder", "{messages}")
])

llm = ChatOpenAI(model = "gpt-3.5-turbo-0125")
# llm = ChatOpenAI(model = "gpt-4o")

router_runnable = router_prompt | llm
coder_runnable = coder_prompt | llm.with_structured_output(CodeResponse)
search_runnable = search_prompt | llm.with_structured_output(SearchResponse)

class Assistant:
  def __init__(self, runnable : Runnable):
    self.runnable = runnable

  def __call__(self, state: State):
    result = self.runnable.invoke(state)
    print("Result :", result)
    next_action = ""
    coderesponse = ""
    output = ""
    if hasattr(result,'content'):
      # print("Setting next_Action")
      next_action = result.content
    else:
      # print("Setting coderesponse")
      if isinstance(result, CodeResponse):
        print('Setting coderesponse')
        coderesponse = result
      else:
        print('Setting output')
        output = result

    if isinstance(result, AIMessage):
      pass
    else:
      result = AIMessage(content=str(result))

    return {"messages": result,
            "next_action": next_action,
            "coderesponse": coderesponse,
            "output":output}

def execute_function_node(state : State):
    # print("State in execute function :", state)
    code = state['coderesponse'].code
    # print("Code :", code)
    # print("Code type :", type(code))
    try:
        # Use subprocess.run to execute the code with capture output
        # print("Executing....")
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, check=True
        )
        # print("Executed..")
        # print("Result :", result)
        # print("Executed and after result..")
        return {"output" : result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        # Handle errors during execution
        print("ERROR ERROR ERROR")
        # print("Error is :", f"Error: {e}")
        error_message = f"Error: {e.stderr.strip()}"
        # print("Error Message :", error_message)
        return {"output": f"Error: {e}"}
    except Exception as e:
    # Handle any other unexpected errors
        print("ERROR ERROR ERROR...")
        return {"output": f"Unexpected Error: {str(e)}"}

def search_or_code(state: State):
  return state["next_action"]

def human_node(state: State):
  pass
  # user_input = input("User : ")
  # return {"messages": HumanMessage(content=user_input)}
  # if user_input == 'y':
  #   graph.update_state(config,as_node="executor")
  #   graph.invoke(None,config,stream_mode="values")
  # return state

def execute_or_not(state: State):
  return state["after_human"]
  # if state["messages"][-1].content == "y":
  #   return "executor"
  # else:
  #   return "router"

graph = StateGraph(State)
graph.add_node("router", Assistant(router_runnable))
graph.add_node("coder", Assistant(coder_runnable))
graph.add_node("searcher", Assistant(search_runnable))
graph.add_node("executor", execute_function_node)
graph.add_node("human", human_node)
graph.add_conditional_edges("router",search_or_code,{"Search":"searcher","Code":"coder"})
# graph.add_edge("coder","executor")
graph.add_edge("coder","human")
graph.add_conditional_edges("human",execute_or_not)
# graph.add_edge("human","executor")
graph.set_entry_point("router")
graph.add_edge("executor",END)
graph.add_edge("searcher",END)

memory = SqliteSaver.from_conn_string(":memory:")

thread_id = str(uuid.uuid4())

config = {"configurable":{
    "thread_id": thread_id
}}

graph = graph.compile(
    checkpointer=memory,
    interrupt_before=["human"]
    )

def run_graph(user_input : str):
    graph_output = graph.invoke(
        {"messages":("user", user_input)},
        config=config,
        stream_mode="values"
    )
    print(graph_output)
    # return graph_output
    coderesponse = graph_output["coderesponse"]
    output = graph_output["output"]
    if coderesponse != '':
      formatted_text = (
        f"**Code:**\n"
        f"{coderesponse.code.strip()}\n\n"
        f"**Summary:**\n"
        f"{coderesponse.summary.replace('1.', '1. ').replace('2.', '2. ').replace('3.', '3. ')}\n\n"
        f"**Risks:**\n"
        f"{coderesponse.risks}"
      )
      return [gr.TextArea(value = formatted_text,visible=True),
              gr.Markdown(visible=True),
              gr.Textbox(visible=True, interactive=True),
              gr.Button(visible=True),
              gr.Textbox(visible=False)]
    elif output != '':
      # return output.response
      return [gr.TextArea(value=output.response,visible=True),
              gr.Markdown(visible=False),
              gr.Textbox(visible=False,interactive=False),
              gr.Button(visible=False),
              gr.Textbox(visible=False)]

def execute_code(user_inp : str):
  if user_inp == 'y':
    human_message = HumanMessage(content=user_inp)
    graph.update_state(
      config,
      {"messages":human_message,"after_human":"executor"},
      as_node="human"
    )
    graph_post_human_output = graph.invoke(
    None,
    config=config,
    stream_mode="values"
    )
    return [gr.Textbox(value = graph_post_human_output["output"],visible=True),
            gr.TextArea(visible=True),
            gr.Markdown(visible=True),
            gr.Textbox(visible=False),
            gr.Button(visible=False)]
    # graph_output1 = graph.invoke(
    #   None,
    #   config=config,
    #   stream_mode="values"
    # )
  else:
    human_message = HumanMessage(content=user_inp)
    print("Human Message :", human_message)
    graph.update_state(
      config,
      {"messages":human_message,"after_human":"router"},
      as_node="human"
    )
    print("Graph State :", graph.get_state(config=config))
    graph_post_human_output = graph.invoke(
      None,
      config=config,
      stream_mode="values"
    )
    print("Graph Output 1 :", graph_post_human_output)
    coderesponse = graph_post_human_output["coderesponse"]
    output = graph_post_human_output["output"]
    if coderesponse != '':
      formatted_text = (
        f"**Code:**\n"
        f"{coderesponse.code.strip()}\n\n"
        f"**Summary:**\n"
        f"{coderesponse.summary.replace('1.', '1. ').replace('2.', '2. ').replace('3.', '3. ')}\n\n"
        f"**Risks:**\n"
        f"{coderesponse.risks}"
      )
      return [gr.Textbox(visible=False),
              gr.TextArea(value = formatted_text, visible=True),
              gr.Markdown(visible=True),
              gr.Textbox(visible=True),
              gr.Button(visible=True)]
    elif output != '':
      # return output.response
      return [gr.Textbox(visible=False),
              gr.TextArea(value = output.response, visible=True),
              gr.Markdown(visible=False),
              gr.Textbox(visible=False),
              gr.Button(visible=False)]
 



# def run_graph_chatbot(user_input : str,history):
#     graph_output = graph.invoke(
#         {"messages":("user", user_input)},
#         config=config,
#         stream_mode="values"
#     )
#     return graph_output

# with gr.Blocks() as demo:
#   Interface = gr.Interface(
#     fn = run_graph,
#     inputs = ["textbox"],
#     outputs=["textarea"]
#   )

# with gr.Blocks() as demo_chat:
#   chat = gr.ChatInterface(
#     fn = run_graph_chatbot,
#   )
  
# if __name__ == "__main__":
#     demo.launch()
