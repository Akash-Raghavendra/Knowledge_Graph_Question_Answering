import os
import json
from retrieval import Retriever
from graph import KnowledgeGraph
from rag_model import GraphRAG

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)              
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PASSAGES_PATH = os.path.join(DATA_DIR, "passages.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

def run_example():
    retriever = Retriever()
    
    if not os.path.exists(retriever.index_path):
        print("[main] FAISS index missing → building index")
        retriever.load_unlabeled_and_artificial()
        retriever.build_and_save_index()
    else:
        print("[main] FAISS index exists → loading index")
        retriever.load_index()

    kg = KnowledgeGraph()
    kg.build_graph_from_passages(retriever.passages)

    rag = GraphRAG()

    question = "Does aspirin reduce the risk of heart attack?"
    retrieved = retriever.retrieve(question, k=5)
    subG = kg.extract_subgraph(retrieved, hop=1)
    graph_summary = kg.textualize_subgraph(subG, max_edges=20)
    for i, p in enumerate(retrieved, 1):
        print(f"[{i}] {p[:320]}{'...' if len(p) > 320 else ''}\n")
    print("Graph summary\n")
    print(graph_summary)
    print("LLM answer")
    ans = rag.answer(question, retrieved, graph_summary)
    print(ans)

def main():
    run_example()

if __name__ == "__main__":
    try:
        main()     
    finally:
        import gc, multiprocessing
        gc.collect()
        try:
            multiprocessing.semaphore_tracker._semaphore_tracker._cleanup()
        except Exception:
            pass