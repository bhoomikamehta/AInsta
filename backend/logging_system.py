import csv
import os
import datetime
from typing import Dict, List, Optional
import json

class ToxicityLogger:
    def __init__(self, log_file_path: str = "logs/toxicity_analysis.csv"):
        """
        Initialize the toxicity analysis logger
        
        Args:
            log_file_path (str): Path to the CSV log file
        """
        self.log_file_path = log_file_path
        self._ensure_log_directory_exists()
        self._initialize_log_file()
    
    def _ensure_log_directory_exists(self):
        """Create log directory if it doesn't exist"""
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def _initialize_log_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file_path):
            headers = [
                'timestamp',
                'original_text',
                'original_toxicity_score',
                'is_toxic',
                'toxicity_threshold',
                'rephrasing_requested',
                'rephrasing_successful',
                'num_rephrases_generated',
                'neutral_rephrase',
                'neutral_toxicity_score',
                'friendly_rephrase', 
                'friendly_toxicity_score',
                'formal_rephrase',
                'formal_toxicity_score',
                'best_improvement',
                'processing_time_seconds',
                'errors',
                'model_used',
                'api_endpoint'
            ]
            
            with open(self.log_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def log_analysis(self, analysis_data: Dict):
        """
        Log a complete toxicity analysis with rephrasing results
        
        Args:
            analysis_data (Dict): Complete analysis results to log
        """
        try:
            # Extract data with safe defaults
            timestamp = datetime.datetime.now().isoformat()
            original_text = analysis_data.get('original_text', '')[:500]  # Truncate long texts
            
            # Original toxicity info
            toxicity_info = analysis_data.get('toxicity', {})
            original_score = toxicity_info.get('score', 0.0)
            is_toxic = toxicity_info.get('is_toxic', False)
            threshold = toxicity_info.get('threshold_used', 0.7)
            
            # Rephrasing info
            rephrases = analysis_data.get('rephrases', [])
            rephrasing_requested = len(rephrases) > 0 or analysis_data.get('include_rephrase', False)
            rephrasing_successful = len(rephrases) > 0
            num_rephrases = len(rephrases)
            
            # Process rephrases by style
            rephrase_data = {
                'neutral': {'text': '', 'score': 0.0},
                'friendly': {'text': '', 'score': 0.0}, 
                'formal': {'text': '', 'score': 0.0}
            }
            
            best_improvement = 0.0
            
            for rephrase in rephrases:
                style = rephrase.get('style', 'neutral')
                if style in rephrase_data:
                    rephrase_data[style]['text'] = rephrase.get('text', '')[:300]
                    rephrase_data[style]['score'] = rephrase.get('toxicity_score', 0.0)
                    
                    # Calculate improvement
                    improvement = original_score - rephrase.get('toxicity_score', original_score)
                    best_improvement = max(best_improvement, improvement)
            
            # Processing info
            processing_time = analysis_data.get('processing_time_seconds', 0.0)
            errors = analysis_data.get('error', '')
            if isinstance(errors, dict):
                errors = json.dumps(errors)
            errors = str(errors)[:200]  # Truncate long error messages
            
            model_used = analysis_data.get('model_used', 'unknown')
            api_endpoint = analysis_data.get('endpoint', 'analyze-comment')
            
            # Prepare row data
            row_data = [
                timestamp,
                original_text,
                original_score,
                is_toxic,
                threshold,
                rephrasing_requested,
                rephrasing_successful,
                num_rephrases,
                rephrase_data['neutral']['text'],
                rephrase_data['neutral']['score'],
                rephrase_data['friendly']['text'],
                rephrase_data['friendly']['score'],
                rephrase_data['formal']['text'], 
                rephrase_data['formal']['score'],
                best_improvement,
                processing_time,
                errors,
                model_used,
                api_endpoint
            ]
            
            # Write to CSV
            with open(self.log_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)
                
        except Exception as e:
            print(f"Error logging analysis: {e}")
            # Log the error itself to a simple error log
            self._log_error(f"Logging failed: {e}", analysis_data)
    
    def _log_error(self, error_message: str, original_data: Dict):
        """Log errors that occur during logging"""
        error_log_path = self.log_file_path.replace('.csv', '_errors.log')
        
        try:
            with open(error_log_path, 'a', encoding='utf-8') as file:
                timestamp = datetime.datetime.now().isoformat()
                file.write(f"{timestamp} - {error_message}\n")
                file.write(f"Original data keys: {list(original_data.keys())}\n")
                file.write("-" * 50 + "\n")
        except Exception:
            pass  # If we can't even log the error, just ignore it
    
    def get_log_stats(self) -> Dict:
        """
        Get basic statistics from the log file
        
        Returns:
            Dict: Statistics about logged analyses
        """
        if not os.path.exists(self.log_file_path):
            return {"error": "No log file found"}
        
        try:
            stats = {
                "total_analyses": 0,
                "toxic_comments": 0,
                "rephrasing_attempts": 0,
                "successful_rephrases": 0,
                "average_toxicity_score": 0.0,
                "average_improvement": 0.0,
                "most_common_errors": []
            }
            
            toxicity_scores = []
            improvements = []
            errors = []
            
            with open(self.log_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    stats["total_analyses"] += 1
                    
                    # Parse toxicity data
                    try:
                        original_score = float(row.get('original_toxicity_score', 0))
                        is_toxic = row.get('is_toxic', '').lower() == 'true'
                        rephrasing_successful = row.get('rephrasing_successful', '').lower() == 'true'
                        best_improvement = float(row.get('best_improvement', 0))
                        
                        toxicity_scores.append(original_score)
                        
                        if is_toxic:
                            stats["toxic_comments"] += 1
                        
                        if row.get('rephrasing_requested', '').lower() == 'true':
                            stats["rephrasing_attempts"] += 1
                        
                        if rephrasing_successful:
                            stats["successful_rephrases"] += 1
                            improvements.append(best_improvement)
                        
                        if row.get('errors'):
                            errors.append(row['errors'])
                            
                    except (ValueError, TypeError):
                        continue  # Skip malformed rows
            
            # Calculate averages
            if toxicity_scores:
                stats["average_toxicity_score"] = sum(toxicity_scores) / len(toxicity_scores)
            
            if improvements:
                stats["average_improvement"] = sum(improvements) / len(improvements)
            
            # Count error types
            error_counts = {}
            for error in errors[:50]:  # Limit to avoid memory issues
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            stats["most_common_errors"] = sorted(
                error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to calculate stats: {e}"}
    
    def export_filtered_logs(self, output_path: str, filters: Dict = None) -> bool:
        """
        Export filtered logs to a new CSV file
        
        Args:
            output_path (str): Path for the exported file
            filters (Dict): Filter criteria (e.g., {'is_toxic': True, 'min_score': 0.8})
        
        Returns:
            bool: Success status
        """
        if not os.path.exists(self.log_file_path):
            return False
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                headers = reader.fieldnames
                
                with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=headers)
                    writer.writeheader()
                    
                    for row in reader:
                        # Apply filters if provided
                        if filters:
                            should_include = True
                            
                            if 'is_toxic' in filters:
                                row_is_toxic = row.get('is_toxic', '').lower() == 'true'
                                if filters['is_toxic'] != row_is_toxic:
                                    should_include = False
                            
                            if 'min_score' in filters:
                                try:
                                    score = float(row.get('original_toxicity_score', 0))
                                    if score < filters['min_score']:
                                        should_include = False
                                except ValueError:
                                    should_include = False
                            
                            if not should_include:
                                continue
                        
                        writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = ToxicityLogger("test_logs.csv")
    
    # Example analysis data
    sample_analysis = {
        'original_text': "You are such an idiot for thinking that!",
        'toxicity': {
            'score': 0.87,
            'is_toxic': True,
            'threshold_used': 0.7
        },
        'rephrases': [
            {
                'style': 'neutral',
                'text': "I disagree with that perspective.",
                'toxicity_score': 0.15
            },
            {
                'style': 'friendly', 
                'text': "I see things differently on this topic.",
                'toxicity_score': 0.08
            },
            {
                'style': 'formal',
                'text': "I respectfully disagree with that viewpoint.",
                'toxicity_score': 0.05
            }
        ],
        'processing_time_seconds': 2.3,
        'model_used': 'llama2',
        'include_rephrase': True
    }
    
    # Log the analysis
    logger.log_analysis(sample_analysis)
    
    # Get stats
    stats = logger.get_log_stats()
    print("Log Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")