import json
from tqdm import tqdm
from knowledge_graph_rag.prompt import knowledge_graph_creation_system_prompt
from knowledge_graph_rag.llm import llm_call
import re
import networkx as nx


class KnowledgeGraphGenerator:
    def create_knowledge_representations(self, tickets):
        knowledge_representations_of_individual_tickets = []
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
                knowledge_representations_of_individual_tickets.append(knowledge_representation)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON. Error: {e}")
                print(f"Problematic response: {response}")

        return knowledge_representations_of_individual_tickets

    def clean_response(self, response):
        response = re.sub(r"```json", "", response)
        response = re.sub(r"```", "", response)
        response = response.strip()
        return response

    def create_knowledge_graph_from_representations(self, representations):
        G = nx.DiGraph()

        def add_edge(source, target, relationship):
            if G.has_edge(source, target):
                G[source][target]["relationship"] += f", {relationship}"
                G[source][target]["weight"] = G[source][target].get("weight", 1) + 1
            else:
                G.add_edge(source, target, relationship=relationship, weight=1)

        for rep in representations:
            for item in rep:
                source = item["entity"]
                if "connections" in item:
                    for conn in item["connections"]:
                        target = conn["entity"]
                        relationship = conn["relationship"]
                        add_edge(source, target, relationship)

        return G
