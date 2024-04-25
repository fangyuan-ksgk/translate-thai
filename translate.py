# from .function_call import *
import anthropic
import pandas as pd
from tqdm import tqdm
from tool_use import parallel_tool_use, parse_tool_use



############
# Claude-3 #
############

# Get Translation from Claude-3
def get_translation_properties():
    properties = {}
    properties["revision"] = {
        "type": "string",
        "description": "Revision of the original thai text to include proper grammar and space, no other changes and addition of text. Provide only Thai revision."
    }
    properties["translation"] = {
        "type": "string",
        "description": "The translation of the revised Thai text to English. Provide only English translation."
    }
    return properties


def get_qa_properties(questions):
    properties = {}
    for i, question in enumerate(questions):
        properties["question_" + str(i) + "_present"] = {
            "type": "boolean",
            "description": "Whether the question of type " + str(i) + " is present in the text. Question Type Description: " + question
        }
        properties["question_" + str(i)] = {
            "type": "string",
            "description": "The question of type " + str(i) + " in the text"
        }
        properties["answer_" + str(i)] = {
            "type": "string",
            "description": "Answer to question " + str(i) 
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


def construct_qa_tool_prompt(tool_name, tool_description, questions):
    properties = get_qa_properties(questions)
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


qa_tool_description = """ 
Parse out the questions and answers according to the specific genre and description. Decide whether the question of that specific genre is present, if it is, provide the specific question and the answer to that. 
"""

translate_tool_description = """
Translate the Thai text to English. Provide only English translation. Also Revise the original thai text to include proper grammar and space, no other changes and addition of text. Provide only Thai revision.
[Example]
สวัสดี --> Hello
"""

def parse_translation_calls(response):
    calls = []
    for content in response.content:
        if content.type=='tool_use' and content.name.startswith('translate_tool'):
            translation = content.input['translation']
            revision = content.input['revision']
            call = {}
            call['name'] = content.name
            call['translation'] = translation
            call['revision'] = revision
            calls.append(call)
    return calls

def parse_qa_calls(response, n_types=5):
    calls = []
    for content in response.content:
        if content.type=='tool_use' and content.name.startswith('qa_tool'):
            call = {}
            call['name'] = content.name
            for i in range(n_types):
                if content.input["question_" + str(i) + "_present"]:
                    call['question_' + str(i)] = content.input['question_' + str(i)]
                    call['answer_' + str(i)] = content.input['answer_' + str(i)]
            calls.append(call)
    return calls

def parse_qa_anthropic(thai_text, questions, api_key):
    tools = []
    tool_name = "qa_tool"
    tool_description = qa_tool_description
    # questions = ["What is the name of the person?", "What is the name of the place?", "What is the name of the thing?", "What is the name of the action?", "What is the name of the time?"]
    tool = construct_qa_tool_prompt(tool_name, tool_description, questions)
    tools.append(tool)

    client = anthropic.Anthropic(api_key = api_key)
    qa_message = {
        "role": "user",
        "content": "Parse out the questions and answers according to the specific genre and description. Here is the text: " + thai_text
    }
    response = client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=512,
        tools=tools,
        messages=[qa_message],
    )
    calls = parse_qa_calls(response)
    return calls

def translate_english_call_anthropic(thai_text, api_key):
    tools = []
    tool_name = "translate_tool"
    tool_description = translate_tool_description
    tool = construct_translation_tool_prompt(tool_name, tool_description)
    tools.append(tool)

    client = anthropic.Anthropic(api_key = api_key)
    translate_message = {
        "role": "user",
        "content": "Translate the Thai text to English. Here is the text: " + thai_text
    }
    response = client.beta.tools.messages.create(
        model="claude-3-opus-20240229",
        # model = "claude-3-haiku-20240229", # "claude-3-opus-20240229
        # model = "claude-3-sonnet-20240229", # "claude-3-opus-20240229
        max_tokens=1024,
        tools=tools,
        messages=[translate_message],
    )
    calls = parse_translation_calls(response)
    return calls


def get_translate(thai_text, api_key):
    num_attempt = 0
    while num_attempt < 3:
        try:
            calls = translate_english_call_anthropic(thai_text, api_key)
            return calls[0]['translation'], calls[0]['revision']
        except:
            num_attempt += 1
    return "NA"


###############
# GPT-4 Turbo #
###############

def contruct_params(properties):
    parameters = {}
    parameters["type"] = "object"
    parameters["properties"] = properties
    parameters["required"] = list(properties.keys())
    return parameters

def construct_tool_prompt(tool_name, tool_description, properties):
    system_prompt = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": contruct_params(properties)
        }
    }
    return system_prompt


def get_translation_gpt(thai_text):

    properties = get_translation_properties()
    tool_name = "translate_tool"
    tool_description = translate_tool_description
    tool = construct_tool_prompt(tool_name, tool_description, properties)

    
    function_calls = parallel_tool_use(name = "translate_tool",
                    instructions = "Revision of the original thai text to include proper grammar and space, revision only in thai. Also provide English translation on the revised thai text.",
                    tools = [tool], 
                    input_text = "Here is the thai text: \n" + thai_text,
                    )

    revise_calls = parse_tool_use(function_calls)

    calls = []
    for revise_call in revise_calls:
        name, arguments = revise_call
        translation = arguments['translation']
        revision = arguments['revision']
        call = {}
        call['name'] = name
        call['translation'] = translation
        call['revision'] = revision
        calls.append(call)
    return calls


def process_file(file_name, api_key):
    df = pd.read_csv(file_name)
    df['thai_transcript'] = df.apply(lambda x: "NA", axis=1)
    df['translation'] = df.apply(lambda x: "NA", axis=1)
    for i in tqdm(range(len(df))):
        text = df['Transcript'].iloc[i]
        translated_text, revision = get_translate(text, api_key)
        df.at[i, 'translation'] = translated_text  
        df.at[i, 'thai_transcript'] = revision 
    df.drop(columns=['English Translation'], inplace=True)
    df.drop(columns=['Transcript'], inplace=True)
    df.to_csv(file_name.replace(".csv", "_llm.csv"), index=False)

