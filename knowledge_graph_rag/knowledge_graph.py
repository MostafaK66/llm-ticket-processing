import json
from tqdm import tqdm
from knowledge_graph_rag.prompt import knowledge_graph_creation_system_prompt
from knowledge_graph_rag.llm import llm_call


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
            response = self.remove_trailing_commas(response)
            knowledge_representations_of_individual_documents.append(
                json.loads(response)
            )

        return knowledge_representations_of_individual_documents

    def remove_trailing_commas(self, json_string):
        return json_string.rstrip(",")
