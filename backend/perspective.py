import os
import requests
from googleapiclient import discovery
import json

class PerspectiveAPI:
    def __init__(self, api_key=None):
        """
        Initialize Perspective API client
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
        
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
    
    def analyze_comment(self, comment_text, attributes=['TOXICITY']):
        """
        Analyze a comment for toxicity and other attributes
        
        Args:
            comment_text (str): The comment to analyze
            attributes (list): List of attributes to analyze. Options include:
                - TOXICITY
                - SEVERE_TOXICITY
                - IDENTITY_ATTACK
                - INSULT
                - PROFANITY
                - THREAT
        
        Returns:
            dict: Analysis results with scores for each attribute
        """
        try:
            # Build the request
            analyze_request = {
                'comment': {'text': comment_text},
                'requestedAttributes': {}
            }
            
            # Add requested attributes
            for attribute in attributes:
                analyze_request['requestedAttributes'][attribute] = {}
            
            # Make the API request
            response = self.client.comments().analyze(body=analyze_request).execute()
            
            # Parse results
            results = {}
            for attribute in attributes:
                if attribute in response['attributeScores']:
                    score_data = response['attributeScores'][attribute]
                    results[attribute.lower()] = {
                        'score': score_data['summaryScore']['value'],
                        'confidence': score_data['summaryScore']['confidence'] if 'confidence' in score_data['summaryScore'] else None
                    }
            
            return {
                'success': True,
                'results': results,
                'original_text': comment_text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_text': comment_text
            }
    
    def is_toxic(self, comment_text, threshold=0.7):
        """
        Simple helper to check if a comment is toxic above a threshold
        
        Args:
            comment_text (str): The comment to check
            threshold (float): Toxicity threshold (0.0 to 1.0)
        
        Returns:
            tuple: (is_toxic: bool, toxicity_score: float)
        """
        result = self.analyze_comment(comment_text)
        
        if not result['success']:
            return False, 0.0
        
        toxicity_score = result['results'].get('toxicity', {}).get('score', 0.0)
        return toxicity_score > threshold, toxicity_score

if __name__ == "__main__":
    
    try:
        perspective = PerspectiveAPI()
        
        # Test with a sample comment
        test_comment = "You are such an idiot!"
        result = perspective.analyze_comment(test_comment)
        
        print("Analysis Result:")
        print(json.dumps(result, indent=2))
        
        # Test the simple is_toxic helper
        is_toxic, score = perspective.is_toxic(test_comment)
        print(f"\nIs toxic: {is_toxic}, Score: {score:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your GOOGLE_API_KEY environment variable")