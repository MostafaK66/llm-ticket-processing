from litellm import completion


def llm_call(messages):
    response = completion(model="gpt-4o", messages=messages)
    return response.choices[0].message.content
