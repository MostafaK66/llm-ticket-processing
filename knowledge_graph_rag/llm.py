from litellm import completion
from knowledge_graph_rag.prompt import detailed_solution_system_prompt, detailed_solution_user_prompt


def llm_call(messages):
    response = completion(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def detailed_solution_query(search_results):
    user_prompt = detailed_solution_user_prompt(search_results)

    messages = [
        {"role": "system", "content": detailed_solution_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return llm_call(messages=messages)
