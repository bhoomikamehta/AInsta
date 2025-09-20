#!/usr/bin/env python3
"""
Debug script to test LLaMA connection and identify issues
"""

import requests
import json
import time

def test_ollama_direct():
    """Test direct connection to Ollama API"""
    print("🔍 Testing direct Ollama connection...")
    
    try:
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama is running (status: {response.status_code})")
        
        # List available models
        models = response.json().get('models', [])
        print(f"📚 Available models: {[model['name'] for model in models]}")
        
        return True, models
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is 'ollama serve' running?")
        return False, []
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False, []

def test_simple_generation(model_name="llama2"):
    """Test simple text generation"""
    print(f"\n🤖 Testing simple generation with {model_name}...")
    
    simple_prompt = "Say hello in one sentence."
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": simple_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "max_tokens": 50
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"✅ Generation successful!")
            print(f"📝 Response: '{generated_text}'")
            return True, generated_text
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False, ""
            
    except requests.exceptions.Timeout:
        print("❌ Generation timed out (30 seconds)")
        return False, ""
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False, ""

def test_rephrasing_prompt(model_name="llama2"):
    """Test the actual rephrasing prompt we use"""
    print(f"\n🔄 Testing rephrasing prompt with {model_name}...")
    
    toxic_comment = "You are an idiot!"
    
    # Use a simpler, more direct prompt
    simple_prompt = f"""Rewrite this comment to be polite and respectful:

Original: "{toxic_comment}"

Polite version:"""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": simple_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 100,
                    "stop": ["\n\n", "Original:"]
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '').strip()
            print(f"✅ Rephrasing successful!")
            print(f"📝 Original: '{toxic_comment}'")
            print(f"✨ Rephrased: '{generated_text}'")
            return True, generated_text
        else:
            print(f"❌ Rephrasing failed with status {response.status_code}")
            print(f"Error response: {response.text}")
            return False, ""
            
    except Exception as e:
        print(f"❌ Rephrasing error: {e}")
        return False, ""

def test_different_models():
    """Test with different available models"""
    print("\n🎯 Testing different models...")
    
    is_connected, models = test_ollama_direct()
    if not is_connected or not models:
        return
    
    # Common model variations to try
    model_variations = [
        "llama2",
        "llama2:7b",
        "llama2:13b", 
        "llama2:latest",
        "tinyllama",
        "phi"
    ]
    
    available_models = [model['name'] for model in models]
    
    for model_var in model_variations:
        # Check if this model variation exists
        matching_models = [m for m in available_models if model_var in m]
        if matching_models:
            model_to_test = matching_models[0]
            print(f"\n🧪 Testing model: {model_to_test}")
            
            success, response = test_simple_generation(model_to_test)
            if success:
                print(f"✅ {model_to_test} is working!")
                return model_to_test
            else:
                print(f"❌ {model_to_test} failed")
    
    return None

def main():
    print("🚀 LLaMA Connection Debugger")
    print("=" * 50)
    
    # Test 1: Basic connection
    is_connected, models = test_ollama_direct()
    if not is_connected:
        print("\n💡 To fix:")
        print("1. Make sure Ollama is installed: https://ollama.ai/download")
        print("2. Start the server: ollama serve")
        print("3. Pull a model: ollama pull llama2")
        return
    
    if not models:
        print("\n💡 No models found. Pull a model first:")
        print("ollama pull llama2")
        return
    
    # Test 2: Find a working model
    working_model = test_different_models()
    if not working_model:
        print("\n❌ No models are responding. Try:")
        print("1. Restart Ollama: killall ollama && ollama serve")
        print("2. Pull a fresh model: ollama pull llama2:latest")
        return
    
    # Test 3: Test our rephrasing logic
    test_rephrasing_prompt(working_model)
    
    print(f"\n🎉 Recommended model to use: {working_model}")
    print("\n💡 If rephrasing still fails in your main app:")
    print("1. Update llama.py to use the working model name")
    print("2. Simplify the rephrasing prompts")
    print("3. Increase timeout values")

if __name__ == "__main__":
    main()