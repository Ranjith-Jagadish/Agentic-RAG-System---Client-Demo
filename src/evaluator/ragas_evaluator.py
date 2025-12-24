"""RAGAs evaluator for RAG system evaluation"""

from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall
)
from src.backend.rag.retriever import RAGRetriever
from src.backend.rag.llm_service import LLMService
from src.backend.agents.crew_orchestrator import CrewOrchestrator
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAs evaluator for RAG system"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.retriever = RAGRetriever()
        self.llm_service = LLMService()
        self.orchestrator = CrewOrchestrator()
        logger.info("RAGAs evaluator initialized")
    
    def generate_evaluation_dataset(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """
        Generate evaluation dataset from questions
        
        Args:
            questions: List of evaluation questions
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dataset for evaluation
        """
        logger.info(f"Generating evaluation dataset with {len(questions)} questions")
        
        results = []
        
        for i, question in enumerate(questions):
            try:
                # Retrieve context
                nodes = self.retriever.retrieve(question)
                contexts = [node.node.text for node in nodes[:3]]  # Top 3 contexts
                
                # Generate answer
                result = self.orchestrator.process_query(question)
                answer = result["response"]
                
                # Get ground truth if provided
                ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else ""
                
                results.append({
                    "question": question,
                    "contexts": contexts,
                    "answer": answer,
                    "ground_truth": ground_truth
                })
                
                logger.info(f"Processed question {i+1}/{len(questions)}")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                continue
        
        # Create dataset
        dataset = Dataset.from_list(results)
        logger.info(f"Created evaluation dataset with {len(results)} samples")
        
        return dataset
    
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system using RAGAs
        
        Args:
            dataset: Evaluation dataset
            metrics: Optional list of metrics (defaults to all)
            
        Returns:
            Evaluation results dictionary
        """
        if metrics is None:
            metrics = [
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall
            ]
        
        logger.info("Starting RAGAs evaluation...")
        
        try:
            # Convert dataset to format expected by RAGAs
            df = dataset.to_pandas()
            
            # Prepare data for evaluation
            eval_data = {
                "question": df["question"].tolist(),
                "contexts": df["contexts"].tolist(),
                "answer": df["answer"].tolist(),
            }
            
            if "ground_truth" in df.columns:
                eval_data["ground_truth"] = df["ground_truth"].tolist()
            
            eval_dataset = Dataset.from_dict(eval_data)
            
            # Run evaluation
            results = evaluate(
                dataset=eval_dataset,
                metrics=metrics
            )
            
            # Convert to dictionary
            results_dict = results.to_pandas().to_dict()
            
            logger.info("Evaluation completed")
            return results_dict
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report
        
        Args:
            results: Evaluation results dictionary
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "RAG System Evaluation Report",
            "=" * 80,
            ""
        ]
        
        # Extract metrics
        if isinstance(results, dict):
            for metric_name, values in results.items():
                if isinstance(values, dict):
                    report_lines.append(f"{metric_name}:")
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {key}: {value:.4f}")
                        else:
                            report_lines.append(f"  {key}: {value}")
                    report_lines.append("")
                else:
                    report_lines.append(f"{metric_name}: {values}")
                    report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report

