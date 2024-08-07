# from litellm import completion
# from knowledge_graph_rag.prompt import detailed_solution_system_prompt, detailed_solution_user_prompt
#
#
# def llm_call(messages):
#     response = completion(model="gpt-4o", messages=messages)
#     return response.choices[0].message.content
#
#
# def detailed_solution_query(search_results):
#     user_prompt = detailed_solution_user_prompt(search_results)
#
#     messages = [
#         {"role": "system", "content": detailed_solution_system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#
#     return llm_call(messages=messages)

from litellm import completion
from knowledge_graph_rag.prompt import detailed_solution_system_prompt, detailed_solution_user_prompt

# Initialize an empty conversation history
conversation_history = []


def llm_call(messages):
    response = completion(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def detailed_solution_query(search_results):
    user_prompt = detailed_solution_user_prompt(search_results)

    # Update conversation history with the new messages
    conversation_history.append({"role": "system", "content": detailed_solution_system_prompt})
    conversation_history.append({"role": "user", "content": user_prompt})

    # Call the LLM with the entire conversation history
    response_content = llm_call(messages=conversation_history)

    # Add the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_content})

    return response_content
