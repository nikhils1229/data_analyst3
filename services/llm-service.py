import os
import openai
from typing import Dict, Any, List, Optional
from utils.logger import Logger

class LLMService:
    """
    Service for interacting with OpenAI's ChatGPT via Chat Completions API
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Initialize OpenAI client if API key is available
        self.openai_client = None
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.logger.info("OpenAI ChatGPT client initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.warning("No OPENAI_API_KEY found in environment variables")
    
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """
        Use ChatGPT to analyze and understand task requirements
        """
        if not self.openai_client:
            return self._fallback_task_analysis(task_description)
        
        try:
            prompt = f"""
            Analyze the following data analysis task and extract key information:
            
            Task: {task_description}
            
            Please provide:
            1. Task type (wikipedia_scraping, database_query, statistical_analysis, visualization, etc.)
            2. Data sources mentioned
            3. Specific questions or requirements
            4. Expected output format
            5. Any special parameters or constraints
            
            Respond in JSON format.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis task parser. Extract key information from task descriptions and respond in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            try:
                import json
                return json.loads(result)
            except json.JSONDecodeError:
                return {"analysis": result, "parsed_by": "chatgpt"}
                
        except Exception as e:
            self.logger.error(f"Error in ChatGPT task analysis: {e}")
            return self._fallback_task_analysis(task_description)
    
    def generate_insights(self, data_summary: str, context: str = "") -> str:
        """
        Generate insights from data analysis results using ChatGPT
        """
        if not self.openai_client:
            return f"Analysis completed. {data_summary}"
        
        try:
            prompt = f"""
            Based on the following data analysis results, provide key insights and interpretations:
            
            Data Summary: {data_summary}
            Context: {context}
            
            Please provide:
            1. Key findings
            2. Notable patterns or trends
            3. Potential implications
            4. Recommendations for further analysis
            
            Keep it concise and actionable.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst providing insights from analysis results. Be concise and focus on actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating insights with ChatGPT: {e}")
            return f"Analysis completed. {data_summary}"
    
    def improve_task_understanding(self, task: str, context: str = "") -> str:
        """
        Use ChatGPT to improve understanding of ambiguous tasks
        """
        if not self.openai_client:
            return task
        
        try:
            prompt = f"""
            The following data analysis task may be ambiguous. 
            Provide a clearer, more specific version while maintaining the original intent:
            
            Original Task: {task}
            Context: {context}
            
            Return an improved task description that:
            1. Clarifies any ambiguous terms
            2. Maintains the original questions and requirements
            3. Specifies expected output format if mentioned
            4. Preserves all URLs and specific instructions
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a task clarification assistant. Help make data analysis tasks more specific while preserving all original requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error improving task understanding: {e}")
            return task
    
    def parse_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language queries into structured analysis parameters using ChatGPT
        """
        if not self.openai_client:
            return self._fallback_query_parsing(query)
        
        try:
            prompt = f"""
            Parse this natural language query into structured analysis parameters:
            
            Query: {query}
            
            Extract and return as JSON:
            {{
                "analysis_type": "count|correlation|regression|comparison|visualization|other",
                "variables": ["list", "of", "variables"],
                "filters": ["any", "conditions"],
                "aggregation": "sum|mean|count|max|min|none",
                "visualization_type": "scatter|bar|line|histogram|none",
                "time_range": "if applicable",
                "specific_values": ["any", "specific", "thresholds"]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a query parser for data analysis tasks. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            try:
                import json
                return json.loads(result)
            except json.JSONDecodeError:
                return {"query": query, "parsed_text": result, "parsed_by": "chatgpt_fallback"}
                
        except Exception as e:
            self.logger.error(f"Error parsing query with ChatGPT: {e}")
            return self._fallback_query_parsing(query)
    
    def _fallback_task_analysis(self, task_description: str) -> Dict[str, Any]:
        """
        Fallback task analysis when ChatGPT is not available
        """
        task_lower = task_description.lower()
        
        analysis = {
            "task_type": "general_analysis",
            "data_sources": [],
            "questions": [],
            "output_format": "json",
            "parsed_by": "fallback_no_chatgpt"
        }
        
        # Simple keyword detection
        if "wikipedia" in task_lower:
            analysis["task_type"] = "wikipedia_scraping"
        elif "duckdb" in task_lower or "parquet" in task_lower:
            analysis["task_type"] = "database_query"
        elif "plot" in task_lower or "chart" in task_lower:
            analysis["visualization_required"] = True
        
        # Extract URLs
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', task_description)
        if urls:
            analysis["data_sources"] = urls
        
        return analysis
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """
        Fallback query parsing when ChatGPT is not available
        """
        query_lower = query.lower()
        
        parsed = {
            "query": query,
            "analysis_type": "unknown",
            "parsed_by": "fallback_no_chatgpt"
        }
        
        # Simple keyword detection
        if "count" in query_lower or "how many" in query_lower:
            parsed["analysis_type"] = "count"
        elif "correlation" in query_lower:
            parsed["analysis_type"] = "correlation"
        elif "plot" in query_lower or "chart" in query_lower:
            parsed["visualization_required"] = True
        
        return parsed
