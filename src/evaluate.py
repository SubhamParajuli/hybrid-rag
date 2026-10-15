import os
import time
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.chain import RAGChain
from src.vectorstore import VectorStoreManager
from src.config import LANGCHAIN_PROJECT, LANGCHAIN_API_KEY


class LangSmithRAGEvaluator:
    def __init__(self):
        if not LANGCHAIN_API_KEY:
            raise ValueError("LANGCHAIN_API_KEY is missing in your .env file.")
        
        self.client = Client()
        self.rag = RAGChain()
        self.vector_manager = VectorStoreManager()
        self.dataset_name = "transformer-rag-ablation-benchmark"

    def setup_dataset(self):
        """Creates or fetches the evaluation dataset in LangSmith."""
        if self.client.has_dataset(dataset_name=self.dataset_name):
            print(f"Dataset '{self.dataset_name}' already exists in LangSmith.")
            return self.client.read_dataset(dataset_name=self.dataset_name)

        print(f"Creating evaluation dataset '{self.dataset_name}' in LangSmith...")
        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="Benchmark dataset evaluating Recall@K, Faithfulness, and Latency across retrieval strategies."
        )

        examples = [
            (
                {"question": "What is the name of the new simple network architecture proposed in the paper?"},
                {"expected_keywords": ["transformer", "attention"], "ground_truth": "The Transformer, based solely on attention mechanisms."}
            ),
            (
                {"question": "What BLEU score did the Transformer achieve on the WMT 2014 English-to-German translation task?"},
                {"expected_keywords": ["28.4", "bleu"], "ground_truth": "Achieved 28.4 BLEU on WMT 2014 English-to-German."}
            ),
            (
                {"question": "What is the computational complexity per layer for self-attention compared to recurrent layers?"},
                {"expected_keywords": ["o(n^2", "self-attention"], "ground_truth": "Self-attention has O(n^2 * d) complexity per layer."}
            )
        ]

        for inputs, outputs in examples:
            self.client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id
            )

        print(f"Dataset created with {len(examples)} examples.")
        return dataset

    def run_evaluation(self):
        """Runs ablation evaluation and logs custom metrics directly to LangSmith."""
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
            "Vector-Search": vector_retriever,
            "BM25-Keyword": bm25_retriever,
            "Hybrid-RRF": hybrid_retriever
        }

        for strategy_name, retriever in retrievers.items():
            print(f"\n🚀 Running LangSmith Experiment for strategy: [{strategy_name}]...")

            # 1. Target function executed for each test case
            def target_predict(inputs: dict) -> dict:
                question = inputs["question"]
                start_time = time.time()
                
                # Fetch retrieved documents
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(question)[:4]
                else:
                    docs = retriever.invoke(question)[:4]
                
                retrieval_latency = (time.time() - start_time) * 1000
                context_str = "\n".join([d.page_content for d in docs])
                
                # Generate answer via RAG chain
                gen_start = time.time()
                answer = self.rag.ask(question)
                gen_latency = (time.time() - gen_start) * 1000
                total_latency = retrieval_latency + gen_latency
                
                return {
                    "output": answer,
                    "context": context_str,
                    "latency_ms": total_latency
                }

            # 2. Custom Python Evaluator for Recall@K
            def eval_recall_at_k(run, example) -> dict:
                expected_kw = example.outputs.get("expected_keywords", [])
                context = run.outputs.get("context", "").lower()
                
                if not expected_kw:
                    return {"key": "recall_at_4", "score": 0.0}
                
                found = sum(1 for kw in expected_kw if kw.lower() in context)
                score = found / len(expected_kw)
                return {"key": "recall_at_4", "score": float(score)}

            # 3. Custom Python Evaluator for Faithfulness (LLM-as-a-judge)
            def eval_faithfulness(run, example) -> dict:
                question = example.inputs.get("question", "")
                context = run.outputs.get("context", "")
                answer = run.outputs.get("output", "")
                
                judge_prompt = f"""You are an objective evaluation judge. Determine if the Answer is strictly supported by the Context.
Question: {question}
Context: {context}
Answer: {answer}
Is the answer fully supported by the context without introducing outside information? Reply with ONLY 'YES' or 'NO'."""
                
                try:
                    judge_res = self.rag.llm.invoke(judge_prompt).content.strip().upper()
                    score = 1.0 if "YES" in judge_res else 0.0
                except Exception:
                    score = 0.5
                    
                return {"key": "faithfulness", "score": float(score)}

            # 4. Custom Python Evaluator for Latency
            def eval_latency(run, example) -> dict:
                latency = run.outputs.get("latency_ms", 0.0)
                return {"key": "latency_ms", "score": float(latency)}

            # Trigger evaluation experiment upload to LangSmith
            evaluate(
                target_predict,
                data=self.dataset_name,
                evaluators=[eval_recall_at_k, eval_faithfulness, eval_latency],
                experiment_prefix=f"ablation-{strategy_name.lower()}",
                metadata={"project_name": LANGCHAIN_PROJECT, "retriever_strategy": strategy_name}
            )

        print("\n✅ All ablation experiments completed and logged to LangSmith with custom metrics!")
        print("🔗 View your metrics breakdown table at: https://smith.langchain.com")


if __name__ == "__main__":
    evaluator = LangSmithRAGEvaluator()
    evaluator.run_evaluation()