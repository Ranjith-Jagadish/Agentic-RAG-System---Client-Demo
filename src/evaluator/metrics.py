"""Evaluation metrics utilities"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def calculate_average_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate average metrics from evaluation results
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Dictionary with average metrics
    """
    averages = {}
    
    for metric_name, values in results.items():
        if isinstance(values, dict):
            numeric_values = [
                v for v in values.values() if isinstance(v, (int, float))
            ]
            if numeric_values:
                averages[metric_name] = sum(numeric_values) / len(numeric_values)
        elif isinstance(values, (int, float)):
            averages[metric_name] = values
    
    return averages


def format_metrics_summary(results: Dict[str, Any]) -> str:
    """
    Format metrics summary for display
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Formatted summary string
    """
    summary_lines = ["Evaluation Metrics Summary:", "-" * 40]
    
    averages = calculate_average_metrics(results)
    
    for metric_name, avg_value in averages.items():
        summary_lines.append(f"{metric_name}: {avg_value:.4f}")
    
    return "\n".join(summary_lines)

