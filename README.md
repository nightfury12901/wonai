# 🧠 Cosmos: Offline AI Assistant Hub

**Cosmos** is a local-first, offline AI assistant combining powerful capabilities like:

- 💬 Conversational AI with LangChain + GPT4All  
- 🎨 Image Generation using ComfyUI  
- 🎬 Video Generation from prompt-based workflows  
- 🔀 Smart Task Routing (text/image/video)  

All done **completely offline** for **privacy**, **speed**, and **control**.

---

## 🚀 Features

### 💬 Conversational AI (GPT4All + LangChain)
- Runs local large language models like Mistral, Nous Hermes, Falcon, etc.
- LangChain-powered routing and chaining.
- No internet or API keys required.

### 🎨 Image Generation (via ComfyUI)
- Uses ComfyUI for Stable Diffusion-based image creation.
- JSON-based workflows to modify or extend.
- Input a prompt, get beautiful AI images locally.

### 🎬 Video Generation
- Chains multiple generated images into smooth video clips.
- Perfect for storytelling, animation, or creative content.
- Fully configurable pipeline via JSON workflows.

### 🔀 Unified AI Router
- A smart backend routes prompts to:
  - GPT4All for text/chat
  - Image generator for visuals
  - Video generator for animations
- You just input a prompt — Cosmos handles the rest.

---

## 🧩 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
