from typing import Dict, Any, List
from utils.logger import Logger
from utils.plot_utils import PlotUtils
import pandas as pd
import numpy as np

class GeneralAnalyzer:
    """
    General purpose analyzer for tasks that don't fit specific categories
    """
    
    def __init__(self):
        self.logger = Logger()
        self.plot_utils = PlotUtils()
    
    def analyze(self, task_description: str, llm_service) -> Dict[str, Any]:
        """
        General analysis method with LLM integration
        """
        try:
            self.logger.log("Starting general analysis")
            
            # Use LLM to understand the task if available
            task_analysis = llm_service.analyze_task(task_description)
            
            # Generate a generic response structure
            result = {
                "message": "Task received and processed",
                "task_analysis": task_analysis,
                "status": "completed",
                "recommendations": [
                    "For Wikipedia scraping tasks, include the Wikipedia URL",
                    "For database queries, specify the exact query or dataset",
                    "For visualization tasks, specify the data source and chart type",
                    "Provide clear questions or requirements for better analysis"
                ]
            }
            
            # If LLM is available, get insights
            if hasattr(llm_service, 'openai_client') and llm_service.openai_client:
                try:
                    insights = llm_service.generate_insights(
                        f"Task: {task_description}",
                        "General analysis request - no specific data processing performed"
                    )
                    result["insights"] = insights
                    result["llm_enhanced"] = True
                except Exception as e:
                    self.logger.warning(f"Could not generate insights: {e}")
                    result["llm_enhanced"] = False
            else:
                result["llm_enhanced"] = False
            
            # Provide more specific guidance based on task content
            task_lower = task_description.lower()
            
            if 'wikipedia' in task_lower:
                result["specific_guidance"] = {
                    "type": "wikipedia_task",
                    "next_steps": [
                        "Ensure Wikipedia URL is accessible",
                        "Specify which table or data to extract",
                        "Define clear questions to answer",
                        "Specify output format (JSON array, JSON object, etc.)"
                    ]
                }
            elif 'duckdb' in task_lower or 'parquet' in task_lower:
                result["specific_guidance"] = {
                    "type": "database_task", 
                    "next_steps": [
                        "Verify S3 bucket access permissions",
                        "Confirm parquet file structure",
                        "Define specific queries to run",
                        "Specify aggregation and filtering requirements"
                    ]
                }
            elif any(word in task_lower for word in ['plot', 'chart', 'graph', 'visualize']):
                result["specific_guidance"] = {
                    "type": "visualization_task",
                    "next_steps": [
                        "Provide data source or sample data",
                        "Specify chart type (scatter, bar, line, etc.)",
                        "Define x and y axes variables",
                        "Specify any special formatting requirements"
                    ]
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in general analysis: {str(e)}")
            return {
                "error": str(e),
                "message": "General analysis failed",
                "status": "error",
                "type": "GeneralAnalysisError"
            }