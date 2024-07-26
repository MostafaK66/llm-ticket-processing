import json
from tqdm import tqdm
from knowledge_graph_rag.prompt import knowledge_graph_creation_system_prompt
from knowledge_graph_rag.llm import llm_call
import re


class KnowledgeGraphGenerator:
    def create_knowledge_representations(self, tickets):
        knowledge_representations_of_individual_documents = []
        for ticket in tqdm(tickets):
            messages = [
                {"role": "system", "content": knowledge_graph_creation_system_prompt},
                {"role": "user", "content": ticket},
            ]

            response = llm_call(messages=messages)

            # Debugging: Print the raw response
            print(f"Raw response: {response}")

            # Ensure the response is lowercased and trailing commas are remove
            response = response.lower()
            response = self.clean_response(response)

            # Debugging: Print the cleaned response
            print(f"Cleaned response: {response}")

            try:
                # Attempt to parse the response as JSON
                knowledge_representation = json.loads(response)
                knowledge_representations_of_individual_documents.append(knowledge_representation)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON. Error: {e}")
                print(f"Problematic response: {response}")

        return knowledge_representations_of_individual_documents

    def clean_response(self, response):
        # Remove Markdown-style code block markers and trailing commas
        response = re.sub(r"```json", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()
        return response
