from huggingface_hub import InferenceClient
import streamlit as st
from dotenv import load_dotenv
import os

# Load API Token
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Initialize the HuggingFace client
client = InferenceClient(token=API_TOKEN)

def generate_text(prompt):
    try:
        result = client.text_generation(
            prompt=prompt,
            model="gpt2",
            max_new_tokens=50
        )
        return result
    except Exception as e:
        return f"Error: {e}"

def test_inference(token):
    print("\n2. Testing Model Inference...")
    client = InferenceClient(token=token)
    
    try:
        # Test text generation with a smaller, more reliable model
        print("Testing text generation...")
        result = client.text_generation(
            prompt="What is artificial intelligence?",
            model="distilgpt2",  # Changed to a smaller model
            max_new_tokens=30,    # Reduced token count
            temperature=0.7
        )
        print("✓ Text generation successful!")
        print(f"• Response: {result}")

        # Test embeddings with a smaller model
        print("\nTesting embeddings...")
        embeddings = client.feature_extraction(
            text="Hello world",
            model="sentence-transformers/paraphrase-MiniLM-L3-v2"  # Smaller model
        )
        print("✓ Embeddings generation successful!")
        print(f"• Embedding shape: {embeddings.shape}")

        return True

    except Exception as e:
        print("✗ Inference error:", str(e))
        print("\nTroubleshooting tips:")
        print("1. The service might be temporarily down - try again in a few minutes")
        print("2. Check if the model is currently available: https://huggingface.co/models")
        print("3. Verify your internet connection")
        print("4. Make sure your API token has inference permissions")
        return False

def main():
    st.title("HuggingFace Text Generation App")
    
    user_input = st.text_input("Enter your prompt:")
    if st.button("Generate"):
        if user_input:
            with st.spinner("Generating text..."):
                output = generate_text(user_input)
            st.success("Generated text:")
            st.write(output)
        else:
            st.error("Please enter a prompt.")

if __name__ == "__main__":
    main()
