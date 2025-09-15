import os
import sys
from perspective import PerspectiveAPI
from llama import LlamaRephrase

def test_perspective_api():
    """Test the Perspective API wrapper"""
    print("=" * 50)
    print("Testing Perspective API")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set!")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        return False
    
    try:
        # Initialize Perspective API
        perspective = PerspectiveAPI(api_key)
        print("✅ Perspective API initialized successfully")
        
        # Test comments
        test_comments = [
            "Hello, how are you today?",  # Non-toxic
            "You are such an idiot!",     # Toxic
            "I disagree with your opinion", # Non-toxic
            "Go kill yourself!",          # Very toxic
            "This product is terrible"    # Mildly negative
        ]
        
        print("\nTesting toxicity detection:")
        print("-" * 30)
        
        for comment in test_comments:
            result = perspective.analyze_comment(comment)
            
            if result['success']:
                toxicity_score = result['results']['toxicity']['score']
                is_toxic = toxicity_score > 0.7
                status = "🔴 TOXIC" if is_toxic else "🟢 CLEAN"
                print(f"{status} [{toxicity_score:.3f}] {comment}")
            else:
                print(f"❌ ERROR: {result['error']} - {comment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Perspective API test failed: {e}")
        return False

def test_llama_service():
    """Test the LLaMA rephrasing wrapper"""
    print("\n" + "=" * 50)
    print("Testing LLaMA Rephrasing Service")
    print("=" * 50)
    
    try:
        # Initialize LLaMA wrapper
        llama = LlamaRephrase()
        
        # Test connection
        is_connected, message = llama.test_connection()
        if not is_connected:
            print(f"❌ LLaMA connection failed: {message}")
            print("\nTo fix this:")
            print("1. Start Ollama: ollama serve")
            print("2. Pull model: ollama pull llama2")
            return False
        
        print(f"✅ LLaMA connection successful: {message}")
        
        # Test rephrasing
        toxic_comments = [
            "You are such an idiot!",
            "This idea is completely stupid and worthless.",
            "I hate this garbage product!",
            "Go away, nobody wants you here!"
        ]
        
        print("\nTesting comment rephrasing:")
        print("-" * 30)
        
        for comment in toxic_comments:
            print(f"\n📝 Original: {comment}")
            
            result = llama.rephrase_comment(comment)
            
            if result['success']:
                print(f"✨ Rephrased: {result['rephrased_text']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLaMA service test failed: {e}")
        return False

def test_end_to_end():
    """Test the complete pipeline"""
    print("\n" + "=" * 50)
    print("Testing End-to-End Pipeline")
    print("=" * 50)
    
    try:
        # Initialize both services
        perspective = PerspectiveAPI()
        llama = LlamaRephrase()
        
        test_comment = "You are such an idiot for thinking that!"
        
        print(f"🔍 Analyzing: {test_comment}")
        
        # Step 1: Check toxicity
        is_toxic, toxicity_score = perspective.is_toxic(test_comment)
        print(f"📊 Toxicity score: {toxicity_score:.3f}")
        print(f"🚨 Is toxic: {is_toxic}")
        
        # Step 2: Rephrase if toxic
        if is_toxic:
            print("\n🔄 Rephrasing toxic comment...")
            rephrase_result = llama.rephrase_comment(test_comment, toxicity_score)
            
            if rephrase_result['success']:
                print(f"✨ Rephrased: {rephrase_result['rephrased_text']}")
                
                # Step 3: Check if rephrased version is less toxic
                _, new_score = perspective.is_toxic(rephrase_result['rephrased_text'])
                print(f"📊 New toxicity score: {new_score:.3f}")
                
                improvement = toxicity_score - new_score
                if improvement > 0:
                    print(f"🎉 Toxicity reduced by {improvement:.3f}!")
                else:
                    print("⚠️  Toxicity not significantly reduced")
            else:
                print(f"❌ Rephrasing failed: {rephrase_result['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Toxicity Detection and Rephrasing System")
    print("=" * 60)
    
    # Test individual components
    perspective_ok = test_perspective_api()
    llama_ok = test_llama_service()
    
    # Test end-to-end only if both components work
    if perspective_ok and llama_ok:
        e2e_ok = test_end_to_end()
        
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print(f"Perspective API: {'✅ OK' if perspective_ok else '❌ FAILED'}")
        print(f"LLaMA Service: {'✅ OK' if llama_ok else '❌ FAILED'}")
        print(f"End-to-End: {'✅ OK' if e2e_ok else '❌ FAILED'}")
        
        if perspective_ok and llama_ok and e2e_ok:
            print("\n🎉 All tests passed! You can now start the API server:")
            print("python main.py")
        else:
            print("\n❌ Some tests failed. Please fix the issues above.")
            sys.exit(1)
    else:
        print("\n❌ Basic components failed. Cannot run end-to-end test.")
        print("\nFix the following:")
        if not perspective_ok:
            print("- Set up Google Perspective API key")
        if not llama_ok:
            print("- Start Ollama service and pull LLaMA model")
        sys.exit(1)

if __name__ == "__main__":
    main()