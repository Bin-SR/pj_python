# -*- coding: utf-8 -*-
"""
04_vlm/llava_inference.py - Using Pretrained LLaVA for VLM Inference

LLaVA (Large Language and Vision Assistant) is a popular open-source VLM.
This module shows how to use a pretrained VLM for inference.

For RTX 3050: use quantized/int8 versions or smaller variants.
  - LLaVA-1.5-7B (full): needs ~16GB VRAM (too large)
  - LLaVA-1.5-7B (4-bit): needs ~6GB VRAM (fits!)
  - MobileVLM: designed for edge devices

For learning, this module demonstrates the API pattern;
actual model loading uses Hugging Face transformers.
"""

import torch


def demo_llava_concept():
    """Demonstrate the LLaVA inference pattern conceptually."""
    print('LLaVA Architecture Overview:')
    print('  1. Vision Encoder (CLIP-ViT-L/14) -> image features')
    print('  2. MLP Projection -> map to LLM embedding space')
    print('  3. Concatenate [IMG tokens] + [TEXT tokens]')
    print('  4. LLM (Vicuna/Llama) -> generate response')

    print('Sample inference code (Hugging Face):')
    code = '''
# pip install transformers accelerate pillow
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

# Load model (use 4-bit for RTX 3050)
model = LlavaForConditionalGeneration.from_pretrained(
    'llava-hf/llava-1.5-7b-hf',
    torch_dtype=torch.float16,
    load_in_4bit=True,  # 4-bit quantization for 6GB VRAM
)
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

# Load image and ask question
image = Image.open('scene.jpg')
conversation = [
    {'role': 'user', 'content': [
        {'type': 'image'},
        {'type': 'text', 'text': 'What objects are on the table?'},
    ]},
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors='pt').to('cuda')
output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
    '''
    print(code)


def rtx3050_recommendations():
    """Hardware-specific recommendations for RTX 3050."""
    print('RTX 3050 VLM Recommendations:')
    print('  1. Use 4-bit quantized models (bitsandbytes)')
    print('  2. Try smaller VLMs: MobileVLM, TinyLLaVA, Bunny')
    print('  3. For training: freeze vision encoder + LLM, train only projection')
    print('  4. Use LoRA for fine-tuning VLMs on custom data')
    print('  5. Offload vision encoder to CPU if needed')


if __name__ == '__main__':
    print('=' * 60)
    print('Pretrained VLM Inference Guide')
    print('=' * 60)
    demo_llava_concept()
    rtx3050_recommendations()
    print('Done!')