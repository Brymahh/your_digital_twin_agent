from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from context import SYSTEM_PROMPT
from tools import handle_tool_calls, tools
from styles import CSS, JS, EXAMPLES


load_dotenv(override=True)

MY_NAME = "Godsgift"
MODEL_NAME = "gpt-5.4-mini"

openai = OpenAI()

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title = f"{MY_NAME}'s Digital Twin",
        description = f"Hi, I'm an assitant that can answer questions about {MY_NAME}'s Career and Work Experience.",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
