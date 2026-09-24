import pickle
from src.lilac.basic_class.graph import Graph, Subgraph, top_level_gcid_by_low_level_gcid

def initiate_graph():
    _parsed_documents_dir = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/parsed_documents/dev"
    _images_dir = "/Users/mytnguyen/Documents/LILaC/datasets/InfoVQA/image_components/dev"
    _subimages_dir = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/image_components_sub/dev"
    _summaries_dir = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/image_summaries/dev"
    _graph_path = "/Users/mytnguyen/Documents/LILaC/artifacts/InfoVQA/components/graph.pickle"
    graph = Graph(
        multimodal_documents_directory = _parsed_documents_dir,
        images_directory    = _images_dir,
        subimages_directory = _subimages_dir,
        summaries_directory = _summaries_dir
    )
    graph.parse_documents()
    with open(_graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"[Retriever] Graph saved to {_graph_path}")

def main():
    return


if __name__ == "__main__":
    initiate_graph()
    