import os
from datetime import datetime

class Logger:
    """
    Simple logging utility for the application
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.log_to_console = True  # Always log to console in serverless
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        
        if self.log_to_console:
            print(formatted_message)
    
    def info(self, message: str):
        """Log info message"""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message"""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message"""
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        """Log debug message"""
        if self.log_level == "DEBUG":
            self.log(message, "DEBUG")
