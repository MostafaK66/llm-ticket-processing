import json
import re
from collections import deque
from knowledge_graph_rag.ticket_graph import ResponseGenerator

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from knowledge_graph_rag.llm import llm_call
from knowledge_graph_rag.prompt import knowledge_graph_creation_system_prompt


class KnowledgeGraphGenerator:
    def __init__(self, transformer_model):
        self.G = None
        self.transformer_model = transformer_model
        self.response_generator = ResponseGenerator(transformer_model)

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

            try:
                knowledge_representation = json.loads(response)
                knowledge_representations_of_individual_tickets.append(
                    knowledge_representation
                )
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

        self.G = G
        return G

    def integrate_embeddings(self, tickets, embeddings):
        node_texts = list(self.G.nodes)
        node_embeddings = self.response_generator.get_embeddings(node_texts)

        for i, ticket in enumerate(tickets):
            issue_match = re.search(r"Issue: (.*?), Solution:", ticket)
            issue = issue_match[1] if issue_match else ticket
            issue_embedding = embeddings[i]

            best_match_node = None
            best_similarity = 0

            for node, node_embedding in zip(self.G.nodes, node_embeddings):
                similarity = cosine_similarity([issue_embedding], [node_embedding])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_node = node

            if best_match_node and best_similarity > 0.7:
                self.G.nodes[best_match_node]["embedding"] = issue_embedding

    def search_ticket(self, input_ticket, input_embedding, max_depth=3):
        knowledge_representations_of_input_ticket = (
            self.create_knowledge_representations(tickets=[input_ticket])
        )
        result = []
        for rep in knowledge_representations_of_input_ticket:
            for item in rep:
                source_entity = item["entity"]
                if source_entity in self.G:
                    result.append(f"\nEntity: {source_entity}")
                    result.extend(self.bfs_traversal(source_entity, max_depth))

        result.append("\nEmbedding-based Similarity Search:")
        if similarity_results := self.embedding_similarity_search(input_embedding):
            result.append(similarity_results[0])

        return "\n".join(result)

    def embedding_similarity_search(self, input_embedding, top_k=5):
        node_embeddings = {
            node: self.G.nodes[node]["embedding"]
            for node in self.G.nodes
            if "embedding" in self.G.nodes[node]
        }
        similarities = []

        for node, embedding in node_embeddings.items():
            similarity = cosine_similarity([input_embedding], [embedding])[0][0]
            similarities.append((node, similarity))

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        return [
            f"Node {node}: Similarity {similarity:.4f}"
            for node, similarity in similarities
        ]

    def bfs_traversal(self, start_node, max_depth):
        visited = set()
        queue = deque([(start_node, 0)])
        result = []
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth:
                break
            if node not in visited:
                visited.add(node)
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        relationship = self.G[node][neighbor]["relationship"]
                        result.append(f"  -> {neighbor} (Relationship: {relationship})")
                        queue.append((neighbor, depth + 1))
        return result
