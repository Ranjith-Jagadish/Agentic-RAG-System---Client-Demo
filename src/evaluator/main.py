"""RAG evaluator console application"""

import argparse
import json
import logging
import sys
from typing import Optional
from pathlib import Path
from src.evaluator.ragas_evaluator import RAGASEvaluator
from src.evaluator.metrics import format_metrics_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_questions(file_path: str) -> list:
    """Load questions from JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "questions" in data:
            return data["questions"]
        else:
            raise ValueError("Invalid JSON format. Expected list or dict with 'questions' key")


def evaluate_rag_system(
    questions_file: str,
    ground_truths_file: Optional[str] = None,
    output_file: Optional[str] = None
) -> None:
    """
    Evaluate the RAG system
    
    Args:
        questions_file: Path to JSON file with questions
        ground_truths_file: Optional path to JSON file with ground truths
        output_file: Optional path to save evaluation report
    """
    try:
        logger.info("Starting RAG system evaluation")
        
        # Load questions
        questions = load_questions(questions_file)
        logger.info(f"Loaded {len(questions)} questions")
        
        # Load ground truths if provided
        ground_truths = None
        if ground_truths_file:
            ground_truths = load_questions(ground_truths_file)
            logger.info(f"Loaded {len(ground_truths)} ground truths")
        
        # Initialize evaluator
        evaluator = RAGASEvaluator()
        
        # Generate evaluation dataset
        logger.info("Generating evaluation dataset...")
        dataset = evaluator.generate_evaluation_dataset(questions, ground_truths)
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.evaluate(dataset)
        
        # Generate report
        logger.info("Generating report...")
        report = evaluator.generate_report(results, output_file)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(format_metrics_summary(results))
        print("\n" + "=" * 80)
        
        if output_file:
            print(f"\nFull report saved to: {output_file}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the evaluator"""
    parser = argparse.ArgumentParser(
        description="RAG System Evaluator using RAGAs"
    )
    parser.add_argument(
        "questions_file",
        type=str,
        help="Path to JSON file containing evaluation questions"
    )
    parser.add_argument(
        "--ground-truths",
        type=str,
        help="Path to JSON file containing ground truth answers"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation report (default: evaluation_report.txt)"
    )
    
    args = parser.parse_args()
    
    output_file = args.output or "evaluation_report.txt"
    
    evaluate_rag_system(
        questions_file=args.questions_file,
        ground_truths_file=args.ground_truths,
        output_file=output_file
    )


if __name__ == "__main__":
    main()

