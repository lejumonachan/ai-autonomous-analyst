import re
import networkx as nx


BIO_TERMS = [
    "cell",
    "tissue",
    "protein",
    "gene",
    "radiation",
    "oxygen",
    "temperature",
    "microgravity",
    "growth",
    "experiment",
    "drug",
    "therapy",
    "oncology",
    "biology",
    "nutrient",
    "bioreactor"
]


def extract_scientific_entities(text):
    entities = []

    text_lower = text.lower()

    for term in BIO_TERMS:
        if term in text_lower:
            entities.append(term)

    return list(set(entities))


def build_knowledge_graph(text):
    entities = extract_scientific_entities(text)

    graph = nx.Graph()

    for entity in entities:
        graph.add_node(entity)

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            graph.add_edge(entities[i], entities[j])

    return graph, entities


def get_graph_edges(graph):
    return list(graph.edges())