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

            response = response.lower()
            response = self.clean_response(response)

            # print(f"Cleaned response: {response}")

            try:

                knowledge_representation = json.loads(response)
                knowledge_representations_of_individual_documents.append(knowledge_representation)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON. Error: {e}")
                print(f"Problematic response: {response}")

        return knowledge_representations_of_individual_documents

    def clean_response(self, response):
        response = re.sub(r"```json", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()
        return response
