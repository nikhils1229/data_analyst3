import os
import json
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs
import cgi
import io

# Import analysis services
from services.task_dispatcher import TaskDispatcher
from services.llm_service import LLMService
from utils.logger import Logger

class handler(BaseHTTPRequestHandler):
    """
    Main Vercel serverless function handler
    """
    
    def __init__(self, *args, **kwargs):
        self.logger = Logger()
        self.llm_service = LLMService()
        self.task_dispatcher = TaskDispatcher(self.llm_service)
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests - return API info"""
        try:
            api_info = {
                'message': 'Data Analyst Agent API',
                'version': '1.0.0',
                'endpoints': {
                    'POST /': 'Submit analysis task via file upload or JSON',
                    'GET /': 'API information'
                },
                'supported_tasks': [
                    'Wikipedia data scraping and analysis',
                    'DuckDB queries on remote datasets',
                    'Statistical analysis and visualization',
                    'Custom data processing tasks'
                ],
                'formats': {
                    'file_upload': 'curl -X POST "https://your-app.vercel.app/api/" -F "file=@question.txt"',
                    'json': 'curl -X POST "https://your-app.vercel.app/api/" -H "Content-Type: application/json" -d \'{"task": "your task"}\''
                }
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(api_info, indent=2).encode('utf-8'))
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def do_POST(self):
        """Handle POST requests - process analysis tasks"""
        try:
            self.logger.log("Received POST request")
            
            # Parse the request
            task_description = self.parse_request()
            
            if not task_description:
                raise ValueError("No task description provided")
            
            self.logger.log(f"Processing task: {task_description[:100]}...")
            
            # Process the task
            start_time = datetime.now()
            result = self.task_dispatcher.process_task(task_description)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.log(f"Task completed in {processing_time:.2f} seconds")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            
        except Exception as e:
            self.logger.log(f"Error processing request: {str(e)}")
            self.logger.log(f"Traceback: {traceback.format_exc()}")
            self.send_error_response(str(e))
    
    def parse_request(self):
        """Parse incoming request to extract task description"""
        content_type = self.headers.get('Content-Type', '')
        content_length = int(self.headers.get('Content-Length', 0))
        
        if content_length == 0:
            return None
        
        # Read the raw data
        raw_data = self.rfile.read(content_length)
        
        if 'multipart/form-data' in content_type:
            # Handle file upload
            return self.parse_multipart_data(raw_data, content_type)
        elif 'application/json' in content_type:
            # Handle JSON data
            data = json.loads(raw_data.decode('utf-8'))
            return data.get('task', '')
        else:
            # Handle plain text
            return raw_data.decode('utf-8')
    
    def parse_multipart_data(self, raw_data, content_type):
        """Parse multipart form data for file uploads"""
        try:
            # Create a file-like object from the raw data
            fp = io.BytesIO(raw_data)
            
            # Parse the multipart data
            environ = {
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': content_type,
                'CONTENT_LENGTH': str(len(raw_data))
            }
            
            form = cgi.FieldStorage(fp=fp, environ=environ)
            
            # Look for the file field
            if 'file' in form:
                file_item = form['file']
                if file_item.file:
                    return file_item.file.read().decode('utf-8')
            
            # Look for other form fields
            for key in form.keys():
                if key != 'file':
                    return form[key].value
                    
            return None
            
        except Exception as e:
            self.logger.log(f"Error parsing multipart data: {e}")
            return None
    
    def send_error_response(self, error_message):
        """Send error response"""
        error_response = {
            'error': error_message,
            'type': 'ProcessingError',
            'timestamp': datetime.now().isoformat()
        }
        
        self.send_response(500)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode('utf-8'))
