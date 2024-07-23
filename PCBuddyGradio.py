import gradio as gr
from PCBuddyNew import execute_graph
from PCBuddyNew1 import run_graph, execute_code

with gr.Blocks() as demo:
    # Interface1 = gr.Interface(fn=execute_graph, inputs="textbox",outputs="textarea")
    interface2 = gr.Textbox(label="User Input",show_label=False,placeholder="Enter your query");
    button = gr.Button(value="Submit")
    interface3 = gr.TextArea(label="Response",visible=False)
    user_message = gr.Markdown(value="Enter 'y' to proceed with execution. If not, provide what else is needed.",
                               visible=False)
    confirmation = gr.Textbox(label="User Response",visible=False)
    confirm_button = gr.Button(visible=False,value="Confirm")
    interface4 = gr.Textbox(label="Final Output",visible=False)
    button.click(run_graph,[interface2],[interface3,
                                         user_message,
                                         confirmation,
                                         confirm_button,
                                         interface4])
    confirm_button.click(execute_code,[confirmation],[interface4,
                                                      interface3,
                                                      user_message,
                                                      confirmation,
                                                      confirm_button,
                                                     ])

if __name__ == "__main__":
    demo.launch()