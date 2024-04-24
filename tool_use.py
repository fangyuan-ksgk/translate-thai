import json, time, openai 

def show_json(obj):
    display(json.loads(obj.model_dump_json()))

def show_json(obj):
    display(json.loads(obj.model_dump_json()))

from openai import OpenAI



def contruct_parameters(properties):
    """
    properties is the desired structured output / argument for the 'function / tool'
    """
    parameters = {}
    parameters["type"] = "object"
    parameters["properties"] = properties
    parameters["required"] = list(properties.keys())
    return parameters


def construct_tool_for_closeAI(tool_name, tool_description, properties):
    """
    A tool is basically a system prompt
    """
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": contruct_parameters(properties)
        }
    }
    return tool


def parallel_tool_use(name, instructions, tools, input_text):

    """
    Parallel Tool Use
    * Similar to a one-shot completion with system prompt and user input
    * Messages dict is not applicable here
    """

    client = OpenAI() # Make you have your API key set in the OPENAI_API_KEY environment variable

    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model="gpt-4-turbo",
        tools=tools,
    )
        
    WEATHER_ASSISTANT_ID = assistant.id

    def submit_message(assistant_id, thread, user_message):
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )
        return client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

    def create_thread_and_run(user_input):
        thread = client.beta.threads.create()
        run = submit_message(WEATHER_ASSISTANT_ID, thread, user_input)
        return thread, run

    def get_response(thread):
        return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

    def wait_on_run(run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run

    # Pretty printing helper
    def pretty_print(messages):
        print("# Messages")
        for m in messages:
            print(f"{m.role}: {m.content[0].text.value}")
        print()

    thread, run = create_thread_and_run(
        input_text
    )
    run = wait_on_run(run, thread)

    return run


def parse_tool_use(run):
    """
    Parse Tool Use Run results
    Function Calls is a sequence of (function name, function arguments) tuples
    """
    function_calls = []
    try:
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)
            function_calls.append((function_name, function_arguments))
            # print(f"Function Name: {function_name}, Arguments: {function_arguments}")
    except:
        print('Error in parsing tool use run.')
        return []
    return function_calls

