# app.py
from flask import Flask, render_template, request, jsonify, Response
from gpt4all import GPT4All
import os
import logging
import threading
import time
import json
from collections import OrderedDict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define available models - using models that are definitely available
MODELS = {
    "mistral": {
        "name": "mistral-7b-instruct-v0.1.Q4_0.gguf",
        "instance": None,
        "loading": False
    },
    "nous-hermes": {
        "name": "nous-hermes-llama2-13b.Q4_0.gguf",
        "instance": None,
        "loading": False
    }
}

# Default model to use
DEFAULT_MODEL = "mistral"
current_model_key = DEFAULT_MODEL

# Simple in-memory LRU cache implementation
class LRUCache:
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        # Move to end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        # If key exists, update value and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        # Add new key-value pair
        self.cache[key] = value
        # If over capacity, remove oldest item (first item)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Initialize response cache
response_cache = LRUCache(capacity=100)

def load_model_async(model_key):
    """Load a model asynchronously"""
    model_info = MODELS[model_key]
    model_info["loading"] = True
    
    try:
        # Optimize model loading with lower memory usage and faster inference
        logging.info(f"Attempting to load model {model_info['name']}...")
        model_info["instance"] = GPT4All(
            model_info['name'], 
            model_path="./models/",
            allow_download=True,
            n_threads=os.cpu_count()  # Use all available CPU threads
        )
        logging.info(f"Model {model_info['name']} loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model {model_info['name']}: {e}")
    finally:
        model_info["loading"] = False

@app.route('/')
def index():
    return render_template('chat.html', models=MODELS, default_model=DEFAULT_MODEL)

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    global current_model_key
    
    # Handle SSE streaming request
    if request.method == 'GET':
        user_message = request.args.get('message', '')
        selected_model = request.args.get('model', current_model_key)
        
        if not user_message:
            return jsonify({'response': 'No message provided'})
            
        model_info = MODELS[selected_model if selected_model in MODELS else current_model_key]
        
        # Check if model is available
        if model_info["instance"] is None or model_info["loading"]:
            return jsonify({
                'response': f"Model {model_info['name']} is not ready. Please try again in a moment."
            })
        
        return Response(
            stream_response(model_info, user_message),
            mimetype='text/event-stream'
        )
    
    # Handle regular POST request
    try:
        data = request.json
        user_message = data.get('message', '')
        selected_model = data.get('model', current_model_key)
        stream = data.get('stream', True)  # Default to streaming
        
        # Update current model if a different one is selected
        if selected_model != current_model_key and selected_model in MODELS:
            current_model_key = selected_model
        
        model_info = MODELS[current_model_key]
        
        if not user_message:
            return jsonify({'response': 'No message provided'})
        
        # Check cache first (only for non-streamed responses)
        if not stream:
            cache_key = f"{model_info['name']}:{user_message}"
            cached_response = response_cache.get(cache_key)
            if cached_response:
                return jsonify({
                    'response': cached_response,
                    'model_used': model_info['name'],
                    'cached': True
                })
        
        # If model is not loaded, start loading it
        if model_info["instance"] is None and not model_info["loading"]:
            thread = threading.Thread(target=load_model_async, args=(current_model_key,))
            thread.start()
            return jsonify({'response': f"Loading {model_info['name']} model. Please try again in a moment."})
        
        # If model is currently loading
        if model_info["loading"]:
            return jsonify({'response': f"Model {model_info['name']} is still loading. Please try again in a moment."})
        
        # If model failed to load
        if model_info["instance"] is None:
            return jsonify({'response': f"Model {model_info['name']} failed to load. Please check the server logs."})
        
        # For regular non-streaming responses
        with model_info["instance"].chat_session():
            response = model_info["instance"].generate(
                user_message,
                max_tokens=1024,
                temp=0.7,          # Lower temperature for more consistent responses
                top_k=40,          # Limit token selection to top 40 options
                top_p=0.9,         # Nucleus sampling
                repeat_penalty=1.1 # Reduce repetition
            )
            
            # Cache the response
            cache_key = f"{model_info['name']}:{user_message}"
            response_cache.put(cache_key, response)
        
        return jsonify({
            'response': response,
            'model_used': model_info['name']
        })
        
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({'response': f'Error: {str(e)}'})

def stream_response(model_info, user_message):
    """Generate and stream the response in chunks"""
    try:
        # Start the generation
        with model_info["instance"].chat_session():
            response_iterator = model_info["instance"].generate(
                user_message,
                max_tokens=1024,
                temp=0.7,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1,
                streaming=True  # Enable streaming
            )
            
            # Stream chunks as they're generated
            full_response = ""
            for chunk in response_iterator:
                full_response += chunk
                # Send each chunk to the client
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Cache the full response for future use
            cache_key = f"{model_info['name']}:{user_message}"
            response_cache.put(cache_key, full_response)
            
            # Send end-of-stream signal
            yield f"data: {json.dumps({'end': True, 'model_used': model_info['name']})}\n\n"
    except Exception as e:
        logging.error(f"Error in streaming response: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/status', methods=['GET'])
def status():
    model_statuses = {}
    for key, model_info in MODELS.items():
        if model_info["instance"] is not None:
            status = "Loaded"
        elif model_info["loading"]:
            status = "Loading"
        else:
            status = "Not loaded"
        
        model_statuses[key] = {
            "name": model_info["name"],
            "status": status,
        }
    
    return jsonify({
        'models': model_statuses,
        'current_model': current_model_key
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model_key
    
    data = request.json
    model_key = data.get('model')
    
    if model_key not in MODELS:
        return jsonify({'status': 'error', 'message': f'Unknown model: {model_key}'})
    
    current_model_key = model_key
    model_info = MODELS[model_key]
    
    # If model is not loaded and not loading, start loading it
    if model_info["instance"] is None and not model_info["loading"]:
        thread = threading.Thread(target=load_model_async, args=(model_key,))
        thread.start()
        return jsonify({
            'status': 'loading', 
            'message': f"Loading {model_info['name']}. This may take a moment."
        })
    
    # If model is loading
    if model_info["loading"]:
        return jsonify({
            'status': 'loading', 
            'message': f"Model {model_info['name']} is already loading. Please wait."
        })
    
    # Model is already loaded
    return jsonify({
        'status': 'success', 
        'message': f"Switched to {model_info['name']}"
    })

@app.route('/preload', methods=['GET'])
def preload_models():
    """Endpoint to preload all models in background"""
    for model_key in MODELS:
        model_info = MODELS[model_key]
        if model_info["instance"] is None and not model_info["loading"]:
            thread = threading.Thread(target=load_model_async, args=(model_key,))
            thread.start()
    
    return jsonify({'status': 'success', 'message': 'Preloading all models'})

if __name__ == '__main__':
    # Make sure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    # Start loading the default model on startup
    thread = threading.Thread(target=load_model_async, args=(DEFAULT_MODEL,))
    thread.start()
    
    app.run(debug=True, threaded=True)