# evaluate.py
import json
import os
from src.retrieval import Retriever
from src.graph import KnowledgeGraph
from src.rag_model import GraphRAG
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tqdm

LABEL_MAP = {0: "yes", 1: "no", 2: "maybe"}

def parse_prediction(text):
    t = (text or "").lower()
    if "yes" in t.split():
        return "yes"
    if "no" in t.split():
        return "no"
    if "maybe" in t:
        return "maybe"
    return "maybe"

def eval():
    if not os.path.exists("data/pubmedqa_labeled.json"):
        raise FileNotFoundError("Please run save_dataset.py first to download labeled data.")

    with open("data/pubmedqa_labeled.json", "r") as f:
        labeled = json.load(f)

    retriever = Retriever()
    try:
        retriever.load_index()
    except FileNotFoundError:
        retriever.load_unlabeled_and_artificial()
        retriever.build_and_save_index()

    kg = KnowledgeGraph()
    kg.build_graph_from_passages(retriever.passages)

    rag = GraphRAG()

    y_true = []
    y_pred = []
    for ex in tqdm.tqdm(labeled, desc="Evaluating"):
        q = ex.get("question", "")
        true_label = LABEL_MAP.get(ex.get("label", 2), "maybe")
        y_true.append(true_label)

        retrieved = retriever.retrieve(q, k=5)
        subG = kg.extract_subgraph(retrieved, hop=1)
        graph_summary = kg.textualize_subgraph(subG, max_edges=20)

        try:
            out = rag.answer(q, retrieved, graph_summary)
        except Exception as e:
            print("Model call failed:", e)
            out = "maybe"

        pred = parse_prediction(out)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=["yes", "no", "maybe"])

    print("\n Evaluation Metrics")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    eval()
