import google.generativeai as genai
import os

# --- Configuration ---
# You need to provide your Google Generative AI API key.
# It's recommended to store your API key securely, for example,
# in an environment variable named 'GOOGLE_API_KEY'.
# Replace 'YOUR_API_KEY' below with your actual key if you are not
# using environment variables (less secure for production).

# Method 1: Get API key from environment variable (Recommended)
API_KEY = os.getenv('GEMINI_API_KEY')

# Method 2: Directly put your API key here (Less Recommended for production)
# API_KEY = 'YOUR_API_KEY' # <--- Replace with your actual API key

if not API_KEY or API_KEY == 'YOUR_API_KEY':
    print("Error: API key not set.")
    print("Please set the GOOGLE_API_KEY environment variable")
    print("or replace 'YOUR_API_KEY' in the script with your actual key.")
    exit()

# Configure the generativeai library with your API key
genai.configure(api_key=API_KEY)

# --- Call ListModels and Print Output ---
print("Fetching list of available models...")

try:
    # list_models() returns an iterable (generator)
    models = genai.list_models()

    print("\nAvailable models:")
    print("--------------------")
    found_models = False
    for model in models:
        found_models = True
        # The model object contains details like name, supported methods, etc.
        # Printing the whole object shows all available information.
        print(model)
        print("--------------------")

    if not found_models:
        print("No models found. This is unexpected.")

except Exception as e:
    print(f"\nAn error occurred while fetching models: {e}")
    print("Please ensure your API key is correct and you have network connectivity.")
