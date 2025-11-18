import json
import os
import faiss
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import re


class Retriever:
    def __init__(self,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 neo4j_uri="",
                 neo4j_user="",
                 neo4j_password=""):

        self.encoder = SentenceTransformer(model_name)

        self.passages = []
        self.index = None

        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.passages_path = os.path.join(ROOT, "data/passages.json")
        self.embeddings_path = os.path.join(ROOT, "data/embeddings.npy")
        self.index_path = os.path.join(ROOT, "data/faiss_index.bin")

        self.kg_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

    def extract_phrases(self, text):
        text = text.lower()
        clean = re.sub(r"[^a-z0-9\s\-]", " ", text)
        words = clean.split()
        if not words:
            return []

        phrases = []
        current = []

        for w in words:
            if len(w) > 2:
                current.append(w)
            else:
                if len(current) > 1:
                    phrases.append(" ".join(current))
                current = []

        if len(current) > 1:
            phrases.append(" ".join(current))

        for w in words:
            if len(w) > 4:
                phrases.append(w)

        return list(set(phrases))


    def get_kg_neighbors(self, term, limit=5):
        with self.kg_driver.session() as session:
            result = session.run(
                """
                MATCH (a:Phrase {text: $term})-[:CO_OCCURS_WITH]-(b)
                RETURN b.text AS neighbor
                LIMIT $limit
                """,
                term=term,
                limit=limit
            )
            return [record["neighbor"] for record in result]


    def retrieve_from_kg(self, query, max_terms=3, per_term=5):
        terms = self.extract_phrases(query)
        terms = terms[:max_terms]  

        kg_results = []

        for t in terms:
            try:
                neighbors = self.get_kg_neighbors(t, limit=per_term)
                for n in neighbors:
                    kg_results.append(f"[KG] {t} ↔ {n}")
            except Exception:
                pass  

        return kg_results


    def retrieve_from_faiss(self, query, k=5):
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        hits = [self.passages[i] for i in I[0]]
        return hits


    def retrieve(self, query, k=5):
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        dense_hits = self.retrieve_from_faiss(query, k)
        kg_hits = self.retrieve_from_kg(query)

        return {
            "dense_passages": dense_hits,
            "kg_expansions": kg_hits,
        }


    def load_unlabeled_and_artificial(self,
                                      unlabeled_path="data/pubmedqa_unlabeled.json",
                                      artificial_path="data/pubmedqa_artificial.json"):
        if not os.path.exists(unlabeled_path) or not os.path.exists(artificial_path):
            raise FileNotFoundError("Run load_data.py to generate files.")

        with open(unlabeled_path, "r") as f:
            unlabeled = json.load(f)
        with open(artificial_path, "r") as f:
            artificial = json.load(f)

        self.passages = [ex.get("context", "") for ex in unlabeled] + \
                        [ex.get("context", "") for ex in artificial]

        os.makedirs("data", exist_ok=True)
        with open(self.passages_path, "w") as pf:
            json.dump(self.passages, pf)

        print("Loaded and saved passages.")


    def build_and_save_index(self):
        if not self.passages:
            raise RuntimeError("No passages loaded.")

        os.makedirs("data", exist_ok=True)

        embeddings = self.encoder.encode(self.passages, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        np.save(self.embeddings_path, embeddings)

        with open(self.passages_path, "w") as pf:
            json.dump(self.passages, pf)

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, self.index_path)

        self.index = index
        print("FAISS embeddings saved.")


    def load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index missing.")
        with open(self.passages_path, "r") as pf:
            self.passages = json.load(pf)
        self.index = faiss.read_index(self.index_path)
        print("Loaded FAISS index.")
