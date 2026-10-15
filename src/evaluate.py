import os
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

from src.chain import RAGChain
from src.config import LANGCHAIN_PROJECT, LANGCHAIN_API_KEY


class RAGEvaluator:
    def __init__(self):
        # Validate LangSmith configuration
        if not LANGCHAIN_API_KEY:
            raise ValueError(
                "LANGCHAIN_API_KEY is missing in your environment variables / .env file. "
                "Please set it to enable LangSmith benchmark tracing."
            )
        
        self.client = Client()
        self.rag = RAGChain()
        self.dataset_name = "hybrid-rag-eval-dataset"

    def create_evaluation_dataset(self):
        """Creates or fetches a test dataset in LangSmith with ground truth QA pairs."""
        if self.client.has_dataset(dataset_name=self.dataset_name):
            print(f"Dataset '{self.dataset_name}' already exists in LangSmith.")
            return self.client.read_dataset(dataset_name=self.dataset_name)

        print(f"Creating evaluation dataset: '{self.dataset_name}'...")
        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="Benchmark dataset for testing Hybrid RAG retrieval accuracy and answer quality."
        )

        # Ground-truth evaluation examples
        examples = [
            (
                {"question": "What is the primary objective of this project?"},
                {"answer": "To build a modular, open-source Hybrid RAG system using LangChain, Chroma, and Ollama."}
            ),
            (
                {"question": "How does the hybrid search retriever combine results?"},
                {"answer": "It combines semantic vector search and BM25 keyword search using Reciprocal Rank Fusion."}
            ),
            (
                {"question": "Where are the vector embeddings stored?"},
                {"answer": "They are stored locally in the Chroma vector database directory."}
            )
        ]

        for inputs, outputs in examples:
            self.client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id
            )

        print(f"Dataset created successfully with {len(examples)} examples.")
        return dataset

    def run_evaluation(self):
        """Runs the benchmark dataset through RAGChain and logs metrics to LangSmith."""
        self.create_evaluation_dataset()

        # 1. Correctness Evaluator
        qa_evaluator = LangChainStringEvaluator("qa")

        # 2. Prediction Target Function
        def predict_rag(inputs: dict) -> dict:
            """Wrapper to pass question and optional chat history to RAGChain."""
            question = inputs["question"]
            chat_history = inputs.get("chat_history", [])  # Handles chat memory if provided
            
            # Execute directly against our chain
            response_text = self.rag.ask(question, chat_history=chat_history)
            return {"output": response_text}

        print("Starting LangSmith evaluation run...")
        results = evaluate(
            predict_rag,
            data=self.dataset_name,
            evaluators=[qa_evaluator],
            experiment_prefix="hybrid-rag-ollama-eval",
            metadata={"project_name": LANGCHAIN_PROJECT}
        )

        print("\nEvaluation complete! View results on your LangSmith dashboard:")
        print("https://smith.langchain.com")
        return results


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()