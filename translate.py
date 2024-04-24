# from .function_call import *
import anthropic
import pandas as pd


############
# Claude-3 #
############

# Get Translation from Claude-3
def get_translation_properties():
    properties = {}
    properties["translation"] = {
        "type": "string",
        "description": "The translation of the Thai text to English. Provide only English translation."
    }
    return properties

def construct_translation_tool_prompt(tool_name, tool_description):
    properties = get_translation_properties()
    argument_names = list(properties.keys())
    system_prompt = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": argument_names,
        },
    }
    return system_prompt

translate_tool_description = """
Translate the Thai text to English. Provide only English translation.
[Example]
สวัสดี --> Hello
"""

def parse_translation_calls(response):
    calls = []
    for content in response.content:
        if content.type=='tool_use' and content.name.startswith('translate_tool'):
            translation = content.input['translation']
            call = {}
            call['name'] = content.name
            call['translation'] = translation
            calls.append(call)
    return calls


def translate_english_call_anthropic(thai_text):
    tools = []
    tool_name = "translate_tool"
    tool_description = translate_tool_description
    tool = construct_translation_tool_prompt(tool_name, tool_description)
    tools.append(tool)

    client = anthropic.Anthropic()
    translate_message = {
        "role": "user",
        "content": "Translate the Thai text to English. Here is the text: " + thai_text
    }
    response = client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        tools=tools,
        messages=[translate_message],
    )
    calls = parse_translation_calls(response)
    return calls


def get_translate(thai_text):
    num_attempt = 0
    while num_attempt < 3:
        try:
            calls = translate_english_call_anthropic(thai_text)
            return calls[0]['translation']
        except:
            num_attempt += 1
    return "NA"
