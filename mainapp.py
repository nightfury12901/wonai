# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import requests
import json
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GENERATED_FOLDER'] = 'static/generated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Make sure upload and generated folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)

# Configure your ComfyUI API URL
COMFY_API_URL = "http://127.0.0.1:8188/api"

@app.route('/')
def index():
    return render_template('gen.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get form data
        prompt = request.form.get('prompt', '')
        generation_type = request.form.get('type', 'image')
        
        # Check if custom workflow was uploaded
        workflow_file = request.files.get('workflow_file')
        
        if workflow_file:
            # Save the uploaded JSON workflow
            filename = secure_filename(workflow_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            workflow_file.save(filepath)
            
            # Load the workflow JSON
            with open(filepath, 'r') as f:
                workflow_data = json.load(f)
        else:
            # Use default workflow based on generation type
            if generation_type == 'image':
                workflow_data = get_default_image_workflow(prompt)
            else:
                workflow_data = get_default_video_workflow(prompt)
        
        # Send to ComfyUI API
        # Queue the prompt
        queue_response = requests.post(
            f"{COMFY_API_URL}/prompt",
            json={"prompt": workflow_data}
        )
        
        if queue_response.status_code != 200:
            return jsonify({"error": f"Failed to queue prompt: {queue_response.text}"}), 500
        
        prompt_id = queue_response.json()["prompt_id"]
        
        # Poll for completion
        output_files = wait_for_generation(prompt_id)
        
        # Return the results
        return jsonify({
            "status": "success",
            "files": output_files,
            "type": generation_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def wait_for_generation(prompt_id, timeout=300):
    """Wait for generation to complete and return output files"""
    start_time = time.time()
    output_files = []
    
    while time.time() - start_time < timeout:
        # Check history endpoint
        history_response = requests.get(f"{COMFY_API_URL}/history/{prompt_id}")
        
        if history_response.status_code == 200:
            history_data = history_response.json()
            
            # Check if we have outputs
            if prompt_id in history_data and "outputs" in history_data[prompt_id]:
                outputs = history_data[prompt_id]["outputs"]
                
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for image in node_output["images"]:
                            # Get image information
                            filename = image['filename']
                            subfolder = image.get('subfolder', '')
                            
                            # Construct URL for viewing
                            image_url = f"/view?filename={filename}&subfolder={subfolder}&type=output"
                            
                            # Download the image to local storage
                            download_url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                            save_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
                            
                            download_file(download_url, save_path)
                            
                            output_files.append({
                                "type": "image",
                                "url": f"/static/generated/{filename}",
                                "filename": filename
                            })
                    
                    if "gifs" in node_output or "videos" in node_output:
                        # Handle gifs
                        if "gifs" in node_output:
                            for gif in node_output["gifs"]:
                                filename = gif['filename']
                                subfolder = gif.get('subfolder', '')
                                
                                # Construct URL for viewing
                                gif_url = f"/view?filename={filename}&subfolder={subfolder}&type=output"
                                
                                # Download the gif to local storage
                                download_url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                                save_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
                                
                                download_file(download_url, save_path)
                                
                                output_files.append({
                                    "type": "video",
                                    "url": f"/static/generated/{filename}",
                                    "filename": filename
                                })
                        
                        # Handle videos
                        if "videos" in node_output:
                            for video in node_output["videos"]:
                                filename = video['filename']
                                subfolder = video.get('subfolder', '')
                                
                                # Construct URL for viewing
                                video_url = f"/view?filename={filename}&subfolder={subfolder}&type=output"
                                
                                # Download the video to local storage
                                download_url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type=output"
                                save_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
                                
                                download_file(download_url, save_path)
                                
                                output_files.append({
                                    "type": "video",
                                    "url": f"/static/generated/{filename}",
                                    "filename": filename
                                })
                
                # If we found outputs, we're done
                if output_files:
                    break
        
        # Wait before checking again
        time.sleep(2)
    
    if not output_files:
        raise Exception("Generation timed out or failed to produce outputs")
    
    return output_files

def download_file(url, save_path):
    """Download a file from the ComfyUI server to local storage"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def get_default_image_workflow(prompt):
    """Return the provided ComfyUI workflow for image generation with updated prompt"""
    workflow = {
        "3": {
            "inputs": {
                "seed": 359922688740345,
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
                "ckpt_name": "dreamshaper_8.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
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
                "text": "text, watermark",
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
    
    return workflow

def get_default_video_workflow(prompt):
    """Return the provided ComfyUI workflow for video generation with updated prompt"""
    workflow = {
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
                "text": prompt if prompt else "1girl, solo, cherry blossom, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, black eyes, upper body, from side",
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
                "seed": 541979697762587,
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
                "ckpt_name": "dreamshaper_8.safetensors"
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
                "frame_rate": 8,
                "loop_count": 0,
                "filename_prefix": "AnimateDiff/animation",
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
    
    return workflow

@app.route('/view')
def view_file():
    """Proxy to ComfyUI's view endpoint"""
    filename = request.args.get('filename')
    subfolder = request.args.get('subfolder', '')
    file_type = request.args.get('type', 'output')
    
    url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type={file_type}"
    
    response = requests.get(url)
    if response.status_code == 200:
        save_path = os.path.join(app.config['GENERATED_FOLDER'], filename)
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        return send_from_directory(app.config['GENERATED_FOLDER'], filename)
    else:
        return "File not found", 404

@app.route('/view_results')
def view_results():
    files = []
    for filename in os.listdir(app.config['GENERATED_FOLDER']):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.mp4')):
            file_path = os.path.join('generated', filename)
            file_type = 'video' if filename.lower().endswith(('.mp4', '.gif')) else 'image'
            files.append({
                'path': file_path,
                'name': filename,
                'type': file_type
            })
    
    return render_template('results.html', files=files)

if __name__ == '__main__':
    app.run(debug=True)