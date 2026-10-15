import os
import time
from typing import List
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.examples import examples
from src.chain import RAGChain
from src.vectorstore import VectorStoreManager
from src.config import LANGCHAIN_PROJECT, LANGCHAIN_API_KEY

import json




class LangSmithRAGEvaluator:
    def __init__(self):
        if not LANGCHAIN_API_KEY:
            raise ValueError("LANGCHAIN_API_KEY is missing in your .env file.")
        
        self.client = Client()
        self.rag = RAGChain()
        self.vector_manager = VectorStoreManager()
        self.eval_dataset=examples
        self.dataset_name = "transformer-hybrid-rag-eval"
        

    def setup_dataset(self):
        """Forces creation/recreation of the dataset and populates it with your 50+ examples."""
        # Delete existing dataset if you want to force a clean slate
        try:
            if self.client.has_dataset(dataset_name=self.dataset_name):
                print(f"Clearing old dataset '{self.dataset_name}' to ensure clean upload...")
                dataset_to_delete = self.client.read_dataset(dataset_name=self.dataset_name)
                self.client.delete_dataset(dataset_id=dataset_to_delete.id)
        except Exception as e:
            print(f"Note on dataset cleanup: {e}")

        print(f"Creating evaluation dataset '{self.dataset_name}' in LangSmith...")
        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="Full benchmark dataset evaluating 50+ examples."
        )

        if not self.eval_dataset or len(self.eval_dataset) == 0:
            raise ValueError("self.eval_dataset is empty! Make sure you define your 50+ examples in __init__.")

        for item in self.eval_dataset:
            if isinstance(item, tuple):
                inp, out = item
            else:
                inp = {"question": item["question"]}
                out = {"expected_keywords": item["expected_keywords"], "ground_truth": item["ground_truth"]}

            self.client.create_example(
                inputs=inp,
                outputs=out,
                dataset_id=dataset.id
            )

        print(f"Successfully uploaded {len(self.eval_dataset)} examples to LangSmith dataset.")
        return dataset

    def run_evaluation(self):
        """Runs ablation evaluation across Vector, BM25, and Hybrid RRF using the cloud dataset."""
        # 1. Ensure dataset is fully created and populated with your 50+ items first
        self.setup_dataset()

        # Initialize retrievers for ablation comparison
        vector_store = self.vector_manager.get_vector_store()
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        all_docs_texts = vector_store.get()["documents"]
        all_docs = [Document(page_content=t) for t in all_docs_texts]
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 4
        
        hybrid_retriever = self.rag.retriever

        retrievers = {
            "Vector Search": vector_retriever,
            "BM25 Keyword": bm25_retriever,
            "Hybrid RRF": hybrid_retriever
        }

        # Run evaluation experiment for each retriever strategy using the dataset name string
        for strategy_name, retriever in retrievers.items():
            print(f"\n🚀 Running LangSmith Experiment for strategy: [{strategy_name}]...")

            def target_predict(inputs: dict) -> dict:
                question = inputs["question"]
                start_time = time.time()
                
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(question)[:4]
                else:
                    docs = retriever.invoke(question)[:4]
                
                retrieval_latency = (time.time() - start_time) * 1000
                context_str = "\n".join([d.page_content for d in docs])
                
                gen_start = time.time()
                answer = self.rag.ask(question)
                gen_latency = (time.time() - gen_start) * 1000
                total_latency = retrieval_latency + gen_latency
                
                return {
                    "output": answer,
                    "context": context_str,
                    "latency_ms": total_latency
                }

            def eval_recall_at_k(run, example) -> dict:
                expected_kw = example.outputs.get("expected_keywords", [])
                context = run.outputs.get("context", "").lower()
                if not expected_kw:
                    return {"key": "recall_at_4", "score": 0.0}
                found = sum(1 for kw in expected_kw if kw.lower() in context)
                return {"key": "recall_at_4", "score": float(found / len(expected_kw))}

            def eval_faithfulness(run, example) -> dict:
                question = example.inputs.get("question", "")
                context = run.outputs.get("context", "")
                answer = run.outputs.get("output", "")
                prompt = f"""You are an objective evaluation judge. Determine if the Answer is strictly supported by the Context.
Question: {question}
Context: {context}
Answer: {answer}
Is the answer fully supported by the context without introducing outside information or hallucinations?
Reply with ONLY 'YES' or 'NO'."""
                try:
                    response = self.rag.llm.invoke(prompt).content.strip().upper()
                    score = 1.0 if "YES" in response else 0.0
                except Exception:
                    score = 0.5
                return {"key": "faithfulness", "score": float(score)}

            def eval_latency(run, example) -> dict:
                return {"key": "latency_ms", "score": float(run.outputs.get("latency_ms", 0.0))}

            # Pass the dataset name string. This fixes the attribute error because 
            # LangSmith fetches valid example objects containing .dataset_id natively.
            evaluate(
                target_predict,
                data=self.dataset_name,
                evaluators=[eval_recall_at_k, eval_faithfulness, eval_latency],
                experiment_prefix=f"ablation-{strategy_name.lower().replace(' ', '-')}",
                metadata={"project_name": LANGCHAIN_PROJECT, "retriever_strategy": strategy_name}
            )

        print("\n✅ All ablation experiments completed and logged to LangSmith!")

if __name__ == "__main__":
    evaluator = LangSmithRAGEvaluator()
    evaluator.run_evaluation()