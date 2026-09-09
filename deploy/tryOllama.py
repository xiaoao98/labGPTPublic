import requests
import json

# The URL of your Ollama server
ollama_url = "http://127.0.0.1:11434/api/generate"

# The data payload for the request
payload = {
    "model": "qwen2.5:7b",
    "prompt": "Write a short poem about the city of Houston.",
    "stream": False # Set to True for streaming responses
}

try:
    # Send the POST request
    response = requests.post(ollama_url, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes

    # Parse the JSON response
    data = response.json()
    print(data['response'])

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")