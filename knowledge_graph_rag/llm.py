from litellm import completion
from knowledge_graph_rag.prompt import detailed_solution_system_prompt, detailed_solution_user_prompt

conversation_history = []


def llm_call(messages):
    response = completion(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def detailed_solution_query(search_results):
    user_prompt = detailed_solution_user_prompt(search_results)

    conversation_history.append({"role": "system", "content": detailed_solution_system_prompt})
    conversation_history.append({"role": "user", "content": user_prompt})

    response_content = llm_call(messages=conversation_history)

    conversation_history.append({"role": "assistant", "content": response_content})

    return response_content
