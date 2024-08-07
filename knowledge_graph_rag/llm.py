from litellm import completion


def llm_call(messages):
    response = completion(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def detailed_solution_query(search_results):
    # Construct the prompt for the detailed solution
    prompt = f"""
    The following are search results related to a ticket issue:

    {search_results}

    Based on these search results, please provide a detailed and comprehensive solution for the issue.
    """

    messages = [
        {"role": "system",
         "content": "You are an AI assistant specialized in providing detailed and comprehensive solutions based on provided search results."},
        {"role": "user", "content": prompt}
    ]

    response = llm_call(messages=messages)
    return response
