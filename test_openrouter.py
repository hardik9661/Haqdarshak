import os
import requests
from dotenv import load_dotenv

# Load configuration
load_dotenv('.config')

def test_openrouter_connection():
    """Test OpenRouter API connection and available models."""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your-openrouter-api-key-here':
        print("❌ OpenRouter API key not found in .config file")
        return False
    
    print(f"🔑 Using OpenRouter API key: {api_key[:20]}...")
    
    # Test API connection
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test with a simple request
    test_data = {
        "model": "openai/o3-pro",
        "messages": [
            {"role": "user", "content": "Hello! This is a test message."}
        ],
        "max_tokens": 50
    }
    
    try:
        print("🔄 Testing OpenRouter API connection...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OpenRouter API connection successful!")
            print(f"📝 Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def list_available_models():
    """List available models on OpenRouter."""
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OpenRouter API key not found")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print("🔄 Fetching available models...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models on OpenRouter:")
            print("=" * 50)
            
            # Group models by provider
            providers = {}
            for model in models.get('data', []):
                provider = model['id'].split('/')[0]
                if provider not in providers:
                    providers[provider] = []
                providers[provider].append(model['id'])
            
            for provider, model_list in providers.items():
                print(f"\n🔹 {provider.upper()}:")
                for model in model_list[:5]:  # Show first 5 models per provider
                    print(f"   - {model}")
                if len(model_list) > 5:
                    print(f"   ... and {len(model_list) - 5} more")
                    
        else:
            print(f"❌ Error fetching models: {response.status_code}")
            print(f"Error details: {response.text}")
            
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing OpenRouter Configuration")
    print("=" * 40)
    
    # Test connection
    if test_openrouter_connection():
        print("\n" + "=" * 40)
        # List available models
        list_available_models()
        
        print("\n" + "=" * 40)
        print("💡 Recommended models for your application:")
        print("   - openai/gpt-3.5-turbo (current setting)")
        print("   - openai/gpt-4")
        print("   - anthropic/claude-3-haiku (cost-effective)")
        print("   - anthropic/claude-3-sonnet (high quality)")
        print("   - google/gemini-pro")
        
        print("\n✅ Your OpenRouter configuration looks good!")
        print("🚀 You can now use the Scheme Research Tool.")
    else:
        print("\n❌ Please check your OpenRouter API key and try again.") 