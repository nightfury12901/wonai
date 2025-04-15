from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import json
import os
import uuid
import time
from threading import Thread, Lock
import base64
from io import BytesIO
import requests
from werkzeug.utils import secure_filename

# Import from the existing code
from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store jobs and their status
jobs = {}
jobs_lock = Lock()

# -- Model Setup --
models = {}
loaded_models = {}

def load_model(model_name):
    """Lazy load models only when needed"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    if model_name == "mistral":
        model = GPT4All(model="mistral-7b-instruct-v0.1.Q4_0.gguf", backend="llama", allow_download=True)
    elif model_name == "hermes":
        model = GPT4All(model="nous-hermes-llama2-13b.Q4_0.gguf", backend="llama", allow_download=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    loaded_models[model_name] = model
    return model

# -- Prompt Template --
base_prompt = PromptTemplate(
    input_variables=["question"],
    template="<s>[INST] {question} [/INST]"
)

# -- Helper Functions --
def classify_task(text: str) -> list:
    """Classify the task based on the text"""
    lowered = text.lower()
    tasks = []

    if any(word in lowered for word in ["video", "animate", "movie", "footage"]):
        tasks.append("video")
    if any(word in lowered for word in ["image", "poster", "visual", "picture", "scenic", "view"]):
        tasks.append("image")
    if any(word in lowered for word in ["script", "seo", "blog", "story", "creative"]):
        tasks.append("creative")
    if not tasks:
        tasks.append("chat")

    return tasks

def generate_text(question: str, model_name: str = "mistral") -> str:
    """Generate text using the specified model"""
    model = load_model(model_name)
    result = base_prompt.format(question=question) | model
    return result

def generate_image(prompt: str, settings: dict) -> dict:
    """Generate an image using ComfyUI"""
    # Extract settings
    model = settings.get('model', 'dreamshaper_8.safetensors')
    width, height = map(int, settings.get('resolution', '512x512').split('x'))
    
    # Convert model name to filename
    model_filename = {
        'dreamshaper': 'dreamshaper_8.safetensors',
        'deliberate': 'deliberate_v5.safetensors',
        'openjourney': 'openjourney_v4.ckpt',
        'sdxl': 'sd_xl_base_1.0.safetensors'
    }.get(model, 'dreamshaper_8.safetensors')
    
    payload = {
        "prompt": {
            "3": {
                "inputs": {
                    "seed": int(time.time()) % 10000000000,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": [
                        "4",
                        0
                    ],
                    "positive": [
                        "6",
                        0
                    ],
                    "negative": [
                        "7",
                        0
                    ],
                    "latent_image": [
                        "5",
                        0
                    ]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "4": {
                "inputs": {
                    "ckpt_name": model_filename
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": [
                        "4",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "7": {
                "inputs": {
                    "text": "text, watermark, bad quality, blurry",
                    "clip": [
                        "4",
                        1
                    ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "8": {
                "inputs": {
                    "samples": [
                        "3",
                        0
                    ],
                    "vae": [
                        "4",
                        2
                    ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "9": {
                "inputs": {
                    "filename_prefix": f"image_{int(time.time())}",
                    "images": [
                        "8",
                        0
                    ]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "Save Image"
                }
            }
        }
    }
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    try:
        r = requests.post("http://localhost:8188/prompt", json=payload)
        if r.status_code != 200:
            return {"status": "error", "message": f"Image generation failed with status code {r.status_code}: {r.text}"}
        
        # Store the job info
        with jobs_lock:
            jobs[job_id] = {
                "type": "image",
                "status": "processing",
                "prompt": prompt,
                "settings": settings,
                "start_time": time.time(),
                "result": None
            }
        
        # Start a worker thread to check for the result
        Thread(target=check_comfyui_result, args=(job_id, "image")).start()
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Image generation started for: '{prompt}'"
        }
    except Exception as e:
        return {"status": "error", "message": f"Image generation failed: {str(e)}"}

def generate_video(prompt: str, settings: dict) -> dict:
    """Generate a video using ComfyUI + AnimateDiff"""
    # Extract settings
    model = settings.get('model', 'dreamshaper_8.safetensors')
    frames = int(settings.get('frames', 16))
    fps = int(settings.get('fps', 8))
    
    # Convert model name to filename
    model_filename = {
        'dreamshaper': 'dreamshaper_8.safetensors',
        'openjourney': 'openjourney_v4.ckpt',
        'anything': 'anything-v5.safetensors',
        'realistic': 'realisticVisionV60B1_v51VAE.safetensors'
    }.get(model, 'dreamshaper_8.safetensors')
    
    payload = {
        "prompt": {
            "2": {
                "inputs": {
                "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                "title": "Load VAE"
                }
            },
            "3": {
                "inputs": {
                "text": prompt,
                "clip": [
                    "4",
                    0
                ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                "title": "CLIP Text Encode (Prompt)"
                }
            },
            "4": {
                "inputs": {
                "stop_at_clip_layer": -2,
                "clip": [
                    "32",
                    1
                ]
                },
                "class_type": "CLIPSetLastLayer",
                "_meta": {
                "title": "CLIP Set Last Layer"
                }
            },
            "6": {
                "inputs": {
                "text": "(worst quality, low quality: 1.4)",
                "clip": [
                    "4",
                    0
                ]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                "title": "CLIP Text Encode (Prompt)"
                }
            },
            "7": {
                "inputs": {
                "seed": int(time.time()) % 10000000000,
                "steps": 20,
                "cfg": 8,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": [
                    "36",
                    0
                ],
                "positive": [
                    "3",
                    0
                ],
                "negative": [
                    "6",
                    0
                ],
                "latent_image": [
                    "9",
                    0
                ]
                },
                "class_type": "KSampler",
                "_meta": {
                "title": "KSampler"
                }
            },
            "9": {
                "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": frames  # Use the specified number of frames
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                "title": "Empty Latent Image"
                }
            },
            "10": {
                "inputs": {
                "samples": [
                    "7",
                    0
                ],
                "vae": [
                    "2",
                    0
                ]
                },
                "class_type": "VAEDecode",
                "_meta": {
                "title": "VAE Decode"
                }
            },
            "32": {
                "inputs": {
                "ckpt_name": model_filename
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                "title": "Load Checkpoint"
                }
            },
            "36": {
                "inputs": {
                "model_name": "mm_sd_v15_v2.ckpt",
                "beta_schedule": "autoselect",
                "model": [
                    "32",
                    0
                ],
                "context_options": [
                    "39",
                    0
                ],
                "sample_settings": [
                    "40",
                    0
                ]
                },
                "class_type": "ADE_AnimateDiffLoaderGen1",
                "_meta": {
                "title": "AnimateDiff Loader 🎭🅐🅓①"
                }
            },
            "37": {
                "inputs": {
                "frame_rate": fps,  # Use the specified FPS
                "loop_count": 0,
                "filename_prefix": f"video_{int(time.time())}",
                "format": "image/gif",
                "pingpong": False,
                "save_output": True,
                "images": [
                    "10",
                    0
                ]
                },
                "class_type": "VHS_VideoCombine",
                "_meta": {
                "title": "Video Combine 🎥🅥🅗🅢"
                }
            },
            "39": {
                "inputs": {
                "context_length": frames,  # Use the specified number of frames
                "context_overlap": 4,
                "fuse_method": "pyramid",
                "use_on_equal_length": False,
                "start_percent": 0,
                "guarantee_steps": 1
                },
                "class_type": "ADE_StandardStaticContextOptions",
                "_meta": {
                "title": "Context Options◆Standard Static 🎭🅐🅓"
                }
            },
            "40": {
                "inputs": {
                "batch_offset": 0,
                "noise_type": "FreeNoise",
                "seed_gen": "comfy",
                "seed_offset": 0,
                "adapt_denoise_steps": False
                },
                "class_type": "ADE_AnimateDiffSamplingSettings",
                "_meta": {
                "title": "Sample Settings 🎭🅐🅓"
                }
            }
        }
    }
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    try:
        r = requests.post("http://localhost:8188/prompt", json=payload)
        if r.status_code != 200:
            return {"status": "error", "message": f"Video generation failed with status code {r.status_code}: {r.text}"}
        
        # Store the job info
        with jobs_lock:
            jobs[job_id] = {
                "type": "video",
                "status": "processing",
                "prompt": prompt,
                "settings": settings,
                "start_time": time.time(),
                "result": None
            }
        
        # Start a worker thread to check for the result
        Thread(target=check_comfyui_result, args=(job_id, "video")).start()
        
        return {
            "job_id": job_id,
            "status": "processing", 
            "message": f"Video generation started for: '{prompt}'"
        }
    except Exception as e:
        return {"status": "error", "message": f"Video generation failed: {str(e)}"}

def check_comfyui_result(job_id, job_type):
    """Check for ComfyUI result and update job status"""
    # In a real implementation, you would poll ComfyUI's websocket
    # For this example, we'll simulate completion after a delay
    wait_time = 5 if job_type == "image" else 15
    time.sleep(wait_time)
    
    # Get the job
    with jobs_lock:
        if job_id not in jobs:
            return
        
        job = jobs[job_id]
        
        # Generate a filename
        timestamp = int(time.time())
        
        if job_type == "image":
            filename = f"image_{timestamp}.png"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # In a real implementation, ComfyUI would save the actual image
            # For demo, we could either use a placeholder or generate one
            create_placeholder_image(file_path, 512, 512)
        else:  # video
            filename = f"video_{timestamp}.gif"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # In a real implementation, ComfyUI would save the actual video
            # For demo, we could use a placeholder
            create_placeholder_video(file_path, 512, 512)
        
        # Update job status
        job["status"] = "completed"
        job["result"] = {
            "url": f"/static/outputs/{filename}",
            "filename": filename
        }

def create_placeholder_image(filepath, width, height):
    """Create a placeholder image for demo purposes"""
    try:
        # Using requests to get a placeholder image
        response = requests.get(f"https://picsum.photos/{width}/{height}")
        with open(filepath, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        # Fallback to a simple colored image if needed

def create_placeholder_video(filepath, width, height):
    """Create a placeholder video (GIF) for demo purposes"""
    try:
        # For demo, we'll just copy a placeholder gif
        # In a real implementation, you'd generate a proper video
        # Just using a placeholder GIF URL for demo
        response = requests.get("https://media.giphy.com/media/l3vR16pONsV8cKkWk/giphy.gif")
        with open(filepath, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f"Error creating placeholder video: {e}")

# -- Routes --
@app.route('/')
def index():
    """Render the main page"""
    return render_template('final.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    """Process a request from the frontend"""
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    message = data['message']
    tasks = classify_task(message)
    settings = data.get('settings', {})
    
    response = {
        "text": "Here's what I've created for you:",
        "tasks": tasks,
        "jobs": {}
    }
    
    # Process each task type
    if "creative" in tasks or "chat" in tasks:
        model_name = settings.get('textModel', 'mistral')
        script_content = generate_text(message, model_name)
        response["script"] = {
            "content": script_content,
            "model": model_name
        }
    
    if "image" in tasks:
        image_settings = {
            "model": settings.get('imageModel', 'dreamshaper'),
            "resolution": settings.get('imageResolution', '512x512')
        }
        image_job = generate_image(message, image_settings)
        response["jobs"]["image"] = image_job
    
    if "video" in tasks:
        video_settings = {
            "model": settings.get('videoModel', 'dreamshaper'),
            "frames": settings.get('videoFrames', 16),
            "fps": settings.get('frameRate', 8)
        }
        video_job = generate_video(message, video_settings)
        response["jobs"]["video"] = video_job
    
    return jsonify(response)

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a job"""
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        
        job = jobs[job_id]
        return jsonify({
            "type": job["type"],
            "status": job["status"],
            "result": job["result"] if job["status"] == "completed" else None
        })

@app.route('/api/placeholder/<width>/<height>', methods=['GET'])
def placeholder_image(width, height):
    """Generate a placeholder image for UI testing"""
    try:
        response = requests.get(f"https://picsum.photos/{width}/{height}")
        return Response(response.content, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": f"Failed to get placeholder: {str(e)}"}), 500

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
