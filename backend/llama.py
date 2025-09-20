import requests
import json
import time
from typing import List, Dict, Optional
from enum import Enum

class RephrasingStyle(str, Enum):
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"  
    FORMAL = "formal"

class LlamaRephrase:
    def __init__(self, base_url="http://localhost:11434", model_name="llama2"):
        """
        Enhanced LLaMA wrapper for rephrasing toxic comments with multiple styles
        
        Args:
            base_url (str): Ollama API base URL
            model_name (str): Name of the LLaMA model to use
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
    
    def generate_multiple_rephrases(self, toxic_comment: str, toxicity_score: float = None, 
                                  styles: List[RephrasingStyle] = None, max_retries: int = 2) -> Dict:
        """
        Generate multiple rephrases with different styles
        
        Args:
            toxic_comment (str): The original toxic comment
            toxicity_score (float): Optional toxicity score for context
            styles (List[RephrasingStyle]): List of styles to generate
            max_retries (int): Number of retry attempts per style
        
        Returns:
            dict: Results containing multiple rephrased versions
        """
        if styles is None:
            styles = [RephrasingStyle.NEUTRAL, RephrasingStyle.FRIENDLY, RephrasingStyle.FORMAL]
        
        results = {
            'success': True,
            'original_text': toxic_comment,
            'toxicity_score': toxicity_score,
            'rephrases': [],
            'errors': [],
            'model_used': self.model_name
        }
        
        for style in styles:
            try:
                rephrase_result = self._generate_single_rephrase(
                    toxic_comment, toxicity_score, style, max_retries
                )
                
                if rephrase_result['success']:
                    results['rephrases'].append({
                        'style': style.value,
                        'text': rephrase_result['rephrased_text'],
                        'generation_time': rephrase_result.get('generation_time', 0),
                        'attempts': rephrase_result.get('attempt', 1)
                    })
                else:
                    results['errors'].append({
                        'style': style.value,
                        'error': rephrase_result['error']
                    })
            except Exception as e:
                results['errors'].append({
                    'style': style.value,
                    'error': f"Unexpected error: {str(e)}"
                })
        
        # If no rephrases succeeded, mark as failed
        if not results['rephrases']:
            results['success'] = False
            results['fallback_message'] = self._get_fallback_message(toxic_comment)
        
        return results
    
    def _generate_single_rephrase(self, toxic_comment: str, toxicity_score: float, 
                                 style: RephrasingStyle, max_retries: int) -> Dict:
        """Generate a single rephrase with specified style"""
        prompt = self._create_style_specific_prompt(toxic_comment, toxicity_score, style)
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self._get_temperature_for_style(style),
                            "top_p": 0.9,
                            "max_tokens": 150,
                            "stop": ["\n\n", "Original:", "Note:"]
                        }
                    },
                    timeout=25
                )
                
                if response.status_code == 200:
                    result = response.json()
                    rephrased_text = result.get('response', '').strip()
                    rephrased_text = self._clean_response(rephrased_text)
                    
                    if rephrased_text and len(rephrased_text) > 5:
                        return {
                            'success': True,
                            'rephrased_text': rephrased_text,
                            'generation_time': time.time() - start_time,
                            'attempt': attempt + 1
                        }
                
                time.sleep(1)  # Brief delay before retry
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f"Request failed after {max_retries} attempts: {str(e)}"
                    }
                time.sleep(2)
        
        return {
            'success': False,
            'error': f"Failed to generate {style.value} rephrase after {max_retries} attempts"
        }
    
    def _create_style_specific_prompt(self, toxic_comment: str, toxicity_score: float, 
                                    style: RephrasingStyle) -> str:
        """Create style-specific prompts for rephrasing"""
        score_context = f" (toxicity score: {toxicity_score:.2f})" if toxicity_score else ""
        
        style_instructions = {
            RephrasingStyle.NEUTRAL: {
                "tone": "neutral and objective",
                "approach": "Remove offensive language while maintaining a balanced, matter-of-fact tone",
                "example": "factual and unbiased"
            },
            RephrasingStyle.FRIENDLY: {
                "tone": "warm and approachable", 
                "approach": "Transform the message into something positive and constructive",
                "example": "encouraging and supportive"
            },
            RephrasingStyle.FORMAL: {
                "tone": "professional and respectful",
                "approach": "Use formal language appropriate for business or academic settings",
                "example": "polite and structured"
            }
        }
        
        instructions = style_instructions[style]
        
        prompt = f"""Task: Rephrase this comment{score_context} in a {instructions['tone']} style.

Original comment: "{toxic_comment}"

Style: {style.value.title()}
Approach: {instructions['approach']}
Target tone: {instructions['example']}

Requirements:
- Remove all toxic, offensive, or harmful language
- Preserve any valid underlying message or opinion
- Use {instructions['tone']} language throughout
- Keep response concise (1-2 sentences)
- Make it sound natural and human

{style.value.title()} rephrase:"""
        
        return prompt
    
    def _get_temperature_for_style(self, style: RephrasingStyle) -> float:
        """Get appropriate temperature setting for each style"""
        temperatures = {
            RephrasingStyle.FORMAL: 0.2,    # More conservative/predictable
            RephrasingStyle.NEUTRAL: 0.3,   # Balanced
            RephrasingStyle.FRIENDLY: 0.4   # More creative/varied
        }
        return temperatures.get(style, 0.3)
    
    def _clean_response(self, response_text: str) -> str:
        """Enhanced response cleaning"""
        # Remove common prefixes
        prefixes_to_remove = [
            "rephrased comment:", "rephrase:", "alternative:", "better version:",
            "neutral rephrase:", "friendly rephrase:", "formal rephrase:",
            "here's a", "a better way", "instead:", "rephrased:"
        ]
        
        cleaned = response_text.strip()
        
        # Remove prefixes (case insensitive)
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes if entire response is quoted
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        
        # Remove extra whitespace and line breaks
        cleaned = ' '.join(cleaned.split())
        
        # Ensure it ends with proper punctuation
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        return cleaned
    
    def _get_fallback_message(self, toxic_comment: str) -> str:
        """Generate a generic fallback when all rephrasing fails"""
        fallback_messages = [
            "I understand you have concerns about this topic.",
            "Thank you for sharing your perspective.",
            "I appreciate you taking the time to comment.",
            "Your feedback has been noted.",
            "I recognize this is an important issue to you."
        ]
        
        # Simple fallback selection based on comment length
        index = len(toxic_comment) % len(fallback_messages)
        return fallback_messages[index]
    
    # Backward compatibility method
    def rephrase_comment(self, toxic_comment: str, toxicity_score: float = None, 
                        max_retries: int = 3) -> Dict:
        """
        Legacy method for single neutral rephrase (backward compatibility)
        """
        result = self.generate_multiple_rephrases(
            toxic_comment, toxicity_score, 
            [RephrasingStyle.NEUTRAL], max_retries
        )
        
        if result['success'] and result['rephrases']:
            return {
                'success': True,
                'original_text': toxic_comment,
                'rephrased_text': result['rephrases'][0]['text'],
                'toxicity_score': toxicity_score,
                'model_used': self.model_name
            }
        else:
            return {
                'success': False,
                'error': result['errors'][0]['error'] if result['errors'] else "Unknown error",
                'original_text': toxic_comment
            }

# Example usage and testing
if __name__ == "__main__":
    llama = LlamaRephrase()
    
    # Test connection
    is_connected, message = llama.test_connection()
    print(f"Connection test: {message}")
    
    if is_connected:
        test_comment = "You are such an idiot for thinking that!"
        
        print(f"\nOriginal: {test_comment}")
        print("=" * 50)
        
        # Test multiple styles
        result = llama.generate_multiple_rephrases(test_comment, 0.85)
        
        if result['success']:
            for rephrase in result['rephrases']:
                print(f"{rephrase['style'].title()}: {rephrase['text']}")
        else:
            print("All rephrasing failed:")
            for error in result['errors']:
                print(f"  {error['style']}: {error['error']}")
    else:
        print("Please start Ollama and pull the model first.")