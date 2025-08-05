import re
import json
import traceback
from typing import Dict, Any, List, Union
from datetime import datetime

from analyses.wikipedia_analyzer import WikipediaAnalyzer
from analyses.duckdb_analyzer import DuckDBAnalyzer
from analyses.general_analyzer import GeneralAnalyzer
from utils.logger import Logger

class TaskDispatcher:
    """
    Central dispatcher that routes tasks to appropriate analyzers
    """
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.logger = Logger()
        
        # Initialize analyzers
        self.wikipedia_analyzer = WikipediaAnalyzer()
        self.duckdb_analyzer = DuckDBAnalyzer()
        self.general_analyzer = GeneralAnalyzer()
        
        # Task patterns for routing
        self.task_patterns = {
            'wikipedia': [
                r'wikipedia',
                r'scrape.*wiki',
                r'highest.*grossing.*films',
                r'List_of_highest-grossing_films',
                r'https://en\.wikipedia\.org'
            ],
            'duckdb': [
                r'indian.*high.*court',
                r'duckdb',
                r's3://.*parquet',
                r'read_parquet',
                r'judgement.*dataset',
                r'judgments.*ecourts'
            ]
        }
    
    def identify_task_type(self, task_description: str) -> str:
        """
        Identify the type of task based on the description
        """
        task_lower = task_description.lower()
        
        # Check for Wikipedia tasks
        for pattern in self.task_patterns['wikipedia']:
            if re.search(pattern, task_lower, re.IGNORECASE):
                self.logger.log(f"Identified Wikipedia task with pattern: {pattern}")
                return 'wikipedia'
        
        # Check for DuckDB tasks
        for pattern in self.task_patterns['duckdb']:
            if re.search(pattern, task_lower, re.IGNORECASE):
                self.logger.log(f"Identified DuckDB task with pattern: {pattern}")
                return 'duckdb'
        
        # Default to general analysis
        self.logger.log("No specific pattern matched, using general analyzer")
        return 'general'
    
    def process_task(self, task_description: str) -> Union[List, Dict]:
        """
        Process a task and return the appropriate response
        """
        try:
            self.logger.log(f"Processing task: {task_description[:100]}...")
            
            # Use LLM to enhance task understanding if available
            if hasattr(self.llm_service, 'openai_client') and self.llm_service.openai_client:
                try:
                    enhanced_task = self.llm_service.improve_task_understanding(task_description)
                    self.logger.log(f"Enhanced task understanding: {enhanced_task[:100]}...")
                    task_description = enhanced_task
                except Exception as e:
                    self.logger.log(f"LLM enhancement failed, using original: {e}")
            
            # Identify task type
            task_type = self.identify_task_type(task_description)
            self.logger.log(f"Identified task type: {task_type}")
            
            # Route to appropriate analyzer
            if task_type == 'wikipedia':
                return self.wikipedia_analyzer.analyze(task_description)
            elif task_type == 'duckdb':
                return self.duckdb_analyzer.analyze(task_description)
            else:
                return self.general_analyzer.analyze(task_description, self.llm_service)
                
        except Exception as e:
            self.logger.log(f"Error in TaskDispatcher: {str(e)}")
            self.logger.log(f"Traceback: {traceback.format_exc()}")
            
            # Return error in expected format
            return {
                "error": str(e),
                "type": "TaskProcessingError",
                "timestamp": datetime.now().isoformat()
            }
    
    def parse_questions_from_task(self, task_description: str) -> List[str]:
        """
        Extract questions from the task description
        """
        questions = []
        
        # Look for numbered questions
        numbered_pattern = r'\d+\.\s+([^0-9]+?)(?=\d+\.|$)'
        numbered_matches = re.findall(numbered_pattern, task_description, re.DOTALL)
        
        if numbered_matches:
            questions = [q.strip() for q in numbered_matches]
        else:
            # Split by common question indicators
            potential_questions = re.split(r'[.!?]\s*(?=[A-Z])', task_description)
            questions = [q.strip() for q in potential_questions if '?' in q or any(
                keyword in q.lower() for keyword in ['how many', 'which', 'what', 'when', 'where', 'why']
            )]
        
        return questions
    
    def extract_output_format(self, task_description: str) -> str:
        """
        Determine expected output format from task description
        """
        task_lower = task_description.lower()
        
        if 'json array' in task_lower:
            return 'array'
        elif 'json object' in task_lower:
            return 'object'
        elif any(phrase in task_lower for phrase in ['respond with a json', 'return json']):
            # Look for array or object indicators
            if any(indicator in task_lower for indicator in ['[', 'array']):
                return 'array'
            else:
                return 'object'
        
        return 'auto'  # Let analyzer decide