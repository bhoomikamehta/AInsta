import requests
import json
import time

class LlamaRephrase:
    def __init__(self, base_url="http://localhost:11434", model_name="llama2"):
        """
        Initialize LLaMA wrapper for rephrasing toxic comments
        """
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"
    
    def test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if any(self.model_name in name for name in model_names):
                    return True, "Connection successful"
                else:
                    return False, f"Model {self.model_name} not found. Available models: {model_names}"
            else:
                return False, f"Ollama not responding. Status: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to Ollama: {str(e)}"
    
    def rephrase_comment(self, toxic_comment, toxicity_score=None, max_retries=3):
        """
        Rephrase a toxic comment to remove toxicity while preserving meaning
        """
        # Create a detailed prompt for rephrasing
        prompt = self._create_rephrase_prompt(toxic_comment, toxicity_score)
        
        for attempt in range(max_retries):
            try:
                # Make request to Ollama
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for more consistent results
                            "top_p": 0.9,
                            "max_tokens": 200
                        }
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    rephrased_text = result.get('response', '').strip()
                    
                    # Clean up the response
                    rephrased_text = self._clean_response(rephrased_text)
                    
                    if rephrased_text:
                        return {
                            'success': True,
                            'original_text': toxic_comment,
                            'rephrased_text': rephrased_text,
                            'toxicity_score': toxicity_score,
                            'attempt': attempt + 1,
                            'model_used': self.model_name
                        }
                
                # If we get here, the response was empty or invalid
                time.sleep(1)  # Brief delay before retry
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    return {
                        'success': False,
                        'error': f"Request failed after {max_retries} attempts: {str(e)}",
                        'original_text': toxic_comment
                    }
                time.sleep(2)  # Wait longer between retries
        
        return {
            'success': False,
            'error': f"Failed to generate rephrased text after {max_retries} attempts",
            'original_text': toxic_comment
        }
    
    def _create_rephrase_prompt(self, toxic_comment, toxicity_score=None):
        """Create a detailed prompt for rephrasing"""
        score_context = f" (toxicity score: {toxicity_score:.2f})" if toxicity_score else ""
        
        prompt = f"""Task: Rephrase the following comment to remove toxic, offensive, or harmful language while preserving the core meaning and intent.

Original comment{score_context}: "{toxic_comment}"

Guidelines:
- Remove insults, profanity, threats, and offensive language
- Keep the main message or opinion if it's valid
- Use respectful, constructive language
- Be concise and natural
- If the comment has no salvageable meaning, suggest a neutral alternative

Rephrased comment:"""
        
        return prompt
    
    def _clean_response(self, response_text):
        """Clean and format the model's response"""
        # Remove common prefixes the model might add
        prefixes_to_remove = [
            "Rephrased comment:",
            "Here's a rephrased version:",
            "A better way to say this would be:",
            "Rephrased:",
            "Alternative:",
            "Better version:"
        ]
        
        cleaned = response_text.strip()
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes if the entire response is quoted
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        
        return cleaned

if __name__ == "__main__":
    # Initialize the rephraser
    llama = LlamaRephrase()
    
    # Test connection
    is_connected, message = llama.test_connection()
    print(f"Connection test: {message}")
    
    if is_connected:
        # Test rephrasing
        test_comments = [
            "You are such an idiot!",
            "This idea is completely stupid and worthless.",
            "I hate this garbage product!"
        ]
        
        print("\nTesting rephrasing:")
        for comment in test_comments:
            print(f"\nOriginal: {comment}")
            result = llama.rephrase_comment(comment)
            
            if result['success']:
                print(f"Rephrased: {result['rephrased_text']}")
            else:
                print(f"Error: {result['error']}")
    else:
        print("\nPlease start Ollama first:")
        print("1. Run: ollama serve")
        print("2. Make sure you have pulled the model: ollama pull llama2")