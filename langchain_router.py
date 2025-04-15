from flask import Flask, render_template, request, jsonify
from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
import requests
import json
import os

app = Flask(__name__)

# -- Model Setup --
MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gpt4all")
os.makedirs(MODEL_DIR, exist_ok=True)

mistral = GPT4All(model="mistral-7b-instruct-v0.1.Q4_0.gguf", backend="llama", allow_download=True)
hermes = GPT4All(model="nous-hermes-llama2-13b.Q4_0.gguf", backend="llama", allow_download=True)

# -- Prompt Template --
base_prompt = PromptTemplate(
    input_variables=["question"],
    template="<s>[INST] {question} [/INST]"
)

# -- LangChain Chains --
mistral_chain = base_prompt | mistral
hermes_chain = base_prompt | hermes

# -- Default settings --
VIDEO_SETTINGS = {
    "model": "dreamshaper_8.safetensors",
    "animation_model": "mm_sd_v15_v2.ckpt",
    "frame_rate": 8,
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 8,
    "negative_prompt": "text, watermark, (worst quality, low quality: 1.4)"
}

# -- ComfyUI Tools --
def generate_image(prompt: str) -> dict:
    payload = {
        "prompt": {
            "3": {
                "inputs": {
                    "seed": 359922688740345,
                    "steps": VIDEO_SETTINGS["steps"],
                    "cfg": VIDEO_SETTINGS["cfg"],
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
                    "ckpt_name": VIDEO_SETTINGS["model"]
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "Load Checkpoint"
                }
            },
            "5": {
                "inputs": {
                    "width": VIDEO_SETTINGS["width"],
                    "height": VIDEO_SETTINGS["height"],
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "Empty Latent Image"
                }
            },
            "6": {
                "inputs": {
                    "text": prompt if prompt else "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
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
                    "text": VIDEO_SETTINGS["negative_prompt"],
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
                    "filename_prefix": "ComfyUI",
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
    
    try:
        r = requests.post("http://localhost:8188/prompt", json=payload)
        if r.status_code != 200:
            return {"status": "error", "message": f"Image generation failed: {r.text}"}
        
        # Get the generated image from ComfyUI
        return {
            "status": "success",
            "type": "image", 
            "message": f"Image generated successfully for prompt: '{prompt}'",
            "prompt": prompt
        }
    except Exception as e:
        return {"status": "error", "message": f"Image generation failed: {e}"}

def generate_video(prompt: str) -> dict:
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
                "text": VIDEO_SETTINGS["negative_prompt"],
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
                "seed": 541979697762587,
                "steps": VIDEO_SETTINGS["steps"],
                "cfg": VIDEO_SETTINGS["cfg"],
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
                "width": VIDEO_SETTINGS["width"],
                "height": VIDEO_SETTINGS["height"],
                "batch_size": 16
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
                "ckpt_name": VIDEO_SETTINGS["model"]
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                "title": "Load Checkpoint"
                }
            },
            "36": {
                "inputs": {
                "model_name": VIDEO_SETTINGS["animation_model"],
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
                "frame_rate": VIDEO_SETTINGS["frame_rate"],
                "loop_count": 0,
                "filename_prefix": "AnimateDiff/video",
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
                "context_length": 16,
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

    try:
        r = requests.post("http://localhost:8188/prompt", json=payload)
        r.raise_for_status()
        return {
            "status": "success", 
            "type": "video",
            "message": f"Video generated successfully for prompt: '{prompt}'",
            "prompt": prompt
        }
    except Exception as e:
        return {"status": "error", "message": f"Video generation failed: {e}"}

# -- Advanced Task Classifier with NLP --
def classify_task(text: str) -> dict:
    lowered = text.lower()
    
    # Pattern matching for different tasks
    tasks = {
        "video": False,
        "image": False,
        "creative": False,
        "chat": False
    }
    
    # Video detection patterns
    video_patterns = [
        "video", "animate", "movie", "footage", "clip", "animation", "motion", 
        "film", "short film", "second video", "minute video", "recording"
    ]
    
    # Image detection patterns
    image_patterns = [
        "image", "picture", "poster", "visual", "photo", "photograph", "painting",
        "illustration", "drawing", "graphic", "design", "view", "scene", "scenic", "wallpaper"
    ]
    
    # Creative text patterns
    creative_patterns = [
        "script", "story", "write", "narrative", "blog", "article", "essay", "creative",
        "content", "seo", "dialog", "dialogue", "conversation", "scene", "screenplay"
    ]
    
    # Check for video keywords
    for pattern in video_patterns:
        if pattern in lowered:
            tasks["video"] = True
            
    # Check for image keywords
    for pattern in image_patterns:
        if pattern in lowered:
            tasks["image"] = True
            
    # Check for creative keywords
    for pattern in creative_patterns:
        if pattern in lowered:
            tasks["creative"] = True
    
    # If nothing specific is detected, default to chat
    if not any(tasks.values()):
        tasks["chat"] = True
        
    return tasks

# -- Context-aware prompt parser --
def parse_prompt(text: str) -> list:
    # Split by conjunctions and punctuation that might separate tasks
    import re
    
    # First, identify potential task separators
    separators = [" and ", " also ", " plus ", " & ", ", ", "; "]
    segments = [text]
    
    for separator in separators:
        new_segments = []
        for segment in segments:
            parts = segment.split(separator)
            new_segments.extend(parts)
        segments = new_segments
    
    # Filter out empty segments and trim whitespace
    segments = [segment.strip() for segment in segments if segment.strip()]
    
    return segments

# -- Content Generator --
def generate_content(prompt: str) -> dict:
    # Parse the prompt into potential task segments
    segments = parse_prompt(prompt)
    
    results = []
    task_types = classify_task(prompt)
    
    # If we have a "creative" task, generate the creative content first
    if task_types["creative"]:
        try:
            creative_response = hermes_chain.invoke({"question": prompt})
            results.append({
                "status": "success",
                "type": "text",
                "message": creative_response,
                "prompt": prompt
            })
        except Exception as e:
            results.append({
                "status": "error",
                "type": "text",
                "message": f"Failed to generate creative content: {e}",
                "prompt": prompt
            })
    
    # If we have an "image" task, generate images for relevant segments
    if task_types["image"]:
        image_prompt = prompt
        # Try to extract image-specific parts from the prompt
        for segment in segments:
            if any(pattern in segment.lower() for pattern in ["image", "picture", "poster", "visual"]):
                image_prompt = segment
                break
        
        results.append(generate_image(image_prompt))
    
    # If we have a "video" task, generate videos for relevant segments
    if task_types["video"]:
        video_prompt = prompt
        # Try to extract video-specific parts from the prompt
        for segment in segments:
            if any(pattern in segment.lower() for pattern in ["video", "animate", "movie", "footage"]):
                video_prompt = segment
                break
        
        results.append(generate_video(video_prompt))
    
    # If no specific task or we have a "chat" task, provide a general response
    if task_types["chat"] and not any([task_types["creative"], task_types["image"], task_types["video"]]):
        try:
            chat_response = mistral_chain.invoke({"question": prompt})
            results.append({
                "status": "success",
                "type": "text",
                "message": chat_response,
                "prompt": prompt
            })
        except Exception as e:
            results.append({
                "status": "error",
                "type": "text",
                "message": f"Failed to generate chat response: {e}",
                "prompt": prompt
            })
    
    return results

@app.route('/')
def index():
    return render_template('test3.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    user_input = data.get('prompt', '')
    
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400
    
    results = generate_content(user_input)
    return jsonify({"results": results})

@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(VIDEO_SETTINGS)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global VIDEO_SETTINGS
    data = request.json
    
    # Update only valid fields
    for key in VIDEO_SETTINGS:
        if key in data:
            VIDEO_SETTINGS[key] = data[key]
    
    return jsonify({"status": "success", "settings": VIDEO_SETTINGS})

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    global VIDEO_SETTINGS
    VIDEO_SETTINGS = {
        "model": "dreamshaper_8.safetensors",
        "animation_model": "mm_sd_v15_v2.ckpt",
        "frame_rate": 8,
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 8,
        "negative_prompt": "text, watermark, (worst quality, low quality: 1.4)"
    }
    return jsonify({"status": "success", "settings": VIDEO_SETTINGS})

@app.route('/api/models', methods=['GET'])
def get_models():
    # This would ideally query ComfyUI for available models
    # For now, we'll return a sample list
    models = {
        "image_models": [
            "dreamshaper_8.safetensors",
            "realistic_vision_5.0.safetensors",
            "meinamix_11.safetensors",
            "animation_world_V4.safetensors",
            "anythingV5_PrtRE.safetensors"
        ],
        "animation_models": [
            "mm_sd_v15_v2.ckpt",
            "mm_sd_v14.ckpt",
            "mm_sd_v15.ckpt",
            "aniverse_v11.ckpt",
            "animation_model.ckpt"
        ]
    }
    return jsonify(models)

if __name__ == '__main__':
    app.run(debug=True)