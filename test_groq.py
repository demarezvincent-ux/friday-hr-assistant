import requests
import os
import sys

# Try to get the key from environment or streamlit secrets
def get_key():
    # Check environment
    key = os.environ.get("FIXED_GROQ_KEY")
    if key:
        return key
    
    # Check .streamlit/secrets.toml if it exists (standard location)
    secrets_path = os.path.expanduser("~/Documents/Friday/.streamlit/secrets.toml")
    if os.path.exists(secrets_path):
        import toml
        try:
            secrets = toml.load(secrets_path)
            return secrets.get("FIXED_GROQ_KEY")
        except:
            pass
    return None

def test_groq():
    key = get_key()
    if not key:
        print("❌ Groq API Key (FIXED_GROQ_KEY) not found in environment or secrets file.")
        return

    print(f"🔍 Testing Groq API with key ending in ...{key[-4:]}")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Connection Successful!")
            print(f"💬 Response: {response.json()['choices'][0]['message']['content']}")
        elif response.status_code == 429:
            print("❌ Error 429: Too Many Requests (Rate Limit Exceeded).")
            print("This usually means your free tier credits or daily limits are exhausted.")
        elif response.status_code == 401:
            print("❌ Error 401: Unauthorized. Please check if your API key is valid.")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
        # Print rate limit headers if available
        limits = {k: v for k, v in response.headers.items() if k.lower().startswith('x-ratelimit')}
        if limits:
            print("\n📊 Rate Limit Info:")
            for k, v in limits.items():
                print(f"   {k}: {v}")
                
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_groq()
