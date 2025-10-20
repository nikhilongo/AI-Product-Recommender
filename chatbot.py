import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Use the free gemini-pro model
model = genai.GenerativeModel("gemini-2.0-flash")

# Create a chat session
chat = model.start_chat(history=[])

print("🤖 Gemini Chatbot (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break
    try:
        response = chat.send_message(user_input)
        print("Gemini:", response.text)
    except Exception as e:
        print("❌ Error:", e)
