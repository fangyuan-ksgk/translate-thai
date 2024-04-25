# from .function_call import *
import anthropic
import pandas as pd
from tqdm import tqdm
from tool_use import parallel_tool_use, parse_tool_use
import os


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
            "description": f"Whether the question similar to {question} (Question indexed {i}) is present in the text."
        }
        properties["question_" + str(i)] = {
            "type": "string",
            "description": f"The exact question in the transcript which is similar to {question} (Question indexed {i}) in the text"
        }
        properties["answer_" + str(i)] = {
            "type": "string",
            "description": f"The exact answer in the transcript to the question similar to {question} (Question indexed {i}) in the text"
        }
    return properties


# If valid question, query, if direct answer, append
def get_check_transcript_properties(preset_questions, prev_question):
    properties = {}
    for i, question in enumerate(preset_questions):
        properties["is_related_question"] = {
            "type": "boolean",
            "description": f"""Whether the question in the transcript is similar to any of the pre-set questions: 
            Preset Questions: 
            {preset_questions}
            """
        }
        properties["is_answer"] = {
            "type": "boolean",
            "description": f"""Whether the transcript is an answer to the previous question:
            {prev_question}
            """
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


def construct_check_transcript_tool_prompt(tool_name, tool_description, preset_questions, prev_question):
    properties = get_check_transcript_properties(preset_questions, prev_question)
    argument_names = list(properties.keys())
    system_prompt = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": argument_names,
        }
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
Parse out the questions and answers according, where the question is similar to the provided ones. Decide whether the question and answer of that specific type of question is present, if it is, provide the specific question and the answer to that. 
"""

translate_tool_description = """
Translate the Thai text to English. Provide only English translation. Also Revise the original thai text to include proper grammar and space, no other changes and addition of text. Provide only Thai revision.
[Example]
สวัสดี --> Hello
"""

check_transcript_tool_description = """
Check whether the current transcript is a question which is similar to any of the pre-set questions. Or is it an answer to the pre-set question.
"""

def parse_check_transcript_calls(response):
    calls = []
    for content in response.content:
        if content.type=='tool_use' and content.name.startswith('check_transcript_tool'):
            call = {}
            call['name'] = content.name
            call['is_related_question'] = content.input['is_related_question']
            call['is_answer'] = content.input['is_answer']
            calls.append(call)
    return calls

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

def parse_qa_calls(response, n_types=6):
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


def check_transcript_call_anthropic(english_text, preset_questions, prev_question, api_key):
    tools = []
    tool_name = "check_transcript_tool"
    tool_description = check_transcript_tool_description
    tool = construct_check_transcript_tool_prompt(tool_name, tool_description, preset_questions, prev_question)
    tools.append(tool)

    client = anthropic.Anthropic(api_key = api_key)
    check_transcript_message = {
        "role": "user",
        "content": "Check whether the current transcript is a question which is similar to any of the pre-set questions. Or is it an answer to the pre-set question. Here is the text: \n" + str(english_text)
    }
    response = client.beta.tools.messages.create(
        # model="claude-3-opus-20240229",
        model = "claude-3-sonnet-20240229", # "claude-3-opus-20240229
        max_tokens=1024,
        tools=tools,
        messages=[check_transcript_message],
    )
    calls = parse_check_transcript_calls(response)
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
    calls = parse_qa_calls(response, n_types = len(questions))
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
    #df.drop(columns=['English Translation'], inplace=True)
    df.drop(columns=['Transcript'], inplace=True)
    df.to_csv(file_name.replace(".csv", "_llm.csv"), index=False)



# qa_questions = ["""Have you ever been denied insurance, had your insurance premiums increased due to sub standard case, or had changes made to the terms and conditions of your insurance application, reinstatement, or policy renewal by this or any other company?
# """,
# """Have you ever used or been addicted to drugs, narcotics, or controlled substances, or been involved in drug trafficking or drug-related offenses?
# """,
# """Have you ever been diagnosed with or treated for, or been observed by a physician to have AIDS, immunodeficiency, cancer, heart disease, vascular disease, diabetes, hypertension, cerebrovascular disease, lung disease, tuberculosis, asthma, blood disease ,liver disease, kidney disease, SLE, physical disabilities, or not?
# """,
# """ In the past 5 years, have you undergone diagnostic tests for diseases such as X-rays, electrocardiograms, blood tests, or specialized examinations, or have you been recommended by your current physician or alternative medicine physician for any treatment?
# """,
# """Have you ever had or currently have symptoms such as muscle weakness, severe chronic headaches, coughing up blood, chest pain, chronic abdominal pain, vomiting or bloody stools, chronic diarrhea, chronic joint pain, palpable masses, or in the past 6 months experienced unexplained fatigue or weight loss?
# """,
# """Have you ever been denied insurance, had your insurance premiums increased due to sub standard case, or had changes made to your insurance policy, terms, or conditions by any insurance company, including this one?
# """,
# """Have you used or been addicted to drugs, narcotics, or controlled substances, or been involved in drug trafficking, or been convicted of drug-related offenses?
# """,
# """Have you ever been diagnosed, treated, or observed by a physician for diseases such as AIDS (HIV), cancer, heart disease, vascular disease, diabetes, hypertension, cerebrovascular disease, lung disease, tuberculosis, asthma, blood disease ,liver disease, kidney disease, SLE, physical disabilities, or others?
# """,
# """In the past 3 years, have you undergone diagnostic tests such as X-rays, electrocardiograms, blood tests, or specialized examinations, or have you received treatment or consulted with current medical or alternative medicine practitioners? If yes, please provide details.
# """]

qa_questions = [
    "Asking about health condition and medical history",
]

###################
# Post Processing #
###################

name_shift = {"llm_translate": "translation", "llm_revision": "thai_transcript"}

def query_question(df, i, query_dict):
    print("Query Question")
    try:
        english_question = df['llm_translate'].iloc[i]
        thai_question = df['llm_revision'].iloc[i]
    except:
        english_question = df['translation'].iloc[i]
        thai_question = df['thai_transcript'].iloc[i]

    start_time_question = df['Start time'].iloc[i]
    end_time_question = df['End time'].iloc[i]
    query_dict = {}
    query_dict['translate_question'] = english_question
    query_dict['thai_question'] = thai_question
    query_dict['start_time_question'] = start_time_question
    query_dict['end_time_question'] = end_time_question
    return query_dict

def query_answer(df, i, query_dict):
    if 'translate_question' not in query_dict:
        return query_dict
    
    print("Answer found, query answer")
    try:
        english_answer = df['llm_translate'].iloc[i]
        thai_answer = df['llm_revision'].iloc[i]
    except:
        english_answer = df['translation'].iloc[i]
        thai_answer = df['thai_transcript'].iloc[i]

    start_time_answer = df['Start time'].iloc[i]
    end_time_answer = df['End time'].iloc[i]
    query_dict['translate_answer'] = english_answer
    query_dict['thai_answer'] = thai_answer
    query_dict['start_time_answer'] = start_time_answer
    query_dict['end_time_answer'] = end_time_answer
    return query_dict

def slot_answer(complete_list, query_dict):
    if 'translate_answer' in query_dict:
        print("Slow in Answer")
        # complete_list.append(query_dict)
    return complete_list + [query_dict], {}


 # Initialize Temporary info
def post_proc_llm(file_name):
    df = pd.read_csv(file_name)
    complete_list = []
    query_dict = {}
    for i in tqdm(range(len(df))):
        try:
            english_text = df['llm_translate'].iloc[i]
        except:
            english_text = df['translation'].iloc[i]

        preset_questions = (("\n").join(qa_questions)).strip()

        prev_question = None
        if 'translate_question' in query_dict:
            prev_question = query_dict['translate_question']
            # print("Previous Question: ", prev_question)
            
        calls = check_transcript_call_anthropic(english_text, preset_questions = preset_questions, prev_question = prev_question, api_key = os.environ['ANTHROPIC_API_KEY'])
        # Initialize Temporary info
        is_related_question = False
        is_answer = False
        
        try:
            call = calls[0]
            # Check Condition
            if 'is_related_question' in call:
                if call['is_related_question']:
                    is_related_question = True
                if call['is_answer']:
                    is_answer = True
                    # print("Found answer")
            # Slot in question & answer
            if is_related_question:
                query_dict = query_question(df, i, query_dict)
                print("Found related question.... ")
            if is_answer and not is_related_question:
                print("Detect Answer")
                # print("Before sloting in answer: ", query_dict)
                query_dict = query_answer(df, i, query_dict)
                # print("After sloting in answer: ", query_dict)
                # print("Before slotting answer: ")
                # print(complete_list)
                # print(query_answer)
                complete_list, query_dict = slot_answer(complete_list, query_dict)
                # print("After Slotting Answer: ")
                # print(complete_list)
                # print(query_dict)
        except:
            print("No related question, or answer")

    df_proc = pd.DataFrame(complete_list)
    df_proc.to_csv(file_name.replace("llm.csv", "_llm_proc.csv"), index=False)
    return 

