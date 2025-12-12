import os
import re
import json
import base64
from io import BytesIO
from openai import OpenAI

def convert_pil_image_to_base64(image, format="PNG"):
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def round_by_factor(x, factor):
    return round(x / factor) * factor

def ceil_by_factor(x, factor):
    return ((x + factor - 1) // factor) * factor

def floor_by_factor(x, factor):
    return (x // factor) * factor

def smart_resize(height, width, min_pixels=100*28*28, max_pixels=16384*28*28, max_ratio=200, factor=28):
    orig_height, orig_width = height, width
    area = orig_height * orig_width
    
    if area < min_pixels:
        scale = (min_pixels / area) ** 0.5
        new_height = ceil_by_factor(orig_height * scale, factor)
        new_width = ceil_by_factor(orig_width * scale, factor)
    elif area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        new_height = floor_by_factor(orig_height * scale, factor)
        new_width = floor_by_factor(orig_width * scale, factor)
    else:
        new_height = ceil_by_factor(orig_height, factor)
        new_width = ceil_by_factor(orig_width, factor)
    
    ratio = max(new_height, new_width) / min(new_height, new_width)
    if ratio > max_ratio:
        if new_height > new_width:
            new_height = max_ratio * new_width
        else:
            new_width = max_ratio * new_height
    
    new_height = max(factor, new_height)
    new_width = max(factor, new_width)
    
    return round_by_factor(new_height, factor), round_by_factor(new_width, factor)

class UITarsBackend:
    def __init__(self, api_base="https://openrouter.ai/api/v1", model_name=None):
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        self.api_base = api_base
        self.model_name = model_name
        self.client = OpenAI(
            base_url=api_base,
            api_key=self.api_key
        )
    
    def call(self, prompt, image, model_name=None, temperature=0.0, max_tokens=4096, max_retries=3):
        """Call UITars model"""
        model = model_name or self.model_name
        b64 = convert_pil_image_to_base64(image, format="PNG")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                return content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ UITars API call failed after {max_retries} attempts: {e}")
                    return None
                print(f"⚠️ UITars API attempt {attempt + 1} failed, retrying...")
        return None
    
    def parse_pixel_coordinates(self, response, image):
        """
        Parse pixel coordinates returned by UITars
        Supports multiple formats:
        1. Action: click(start_box='(x,y)')
        2. Action: click(point='<point>x y</point>')
        3. Pure coordinates (x, y) or [x, y]
        4. JSON format {"coordinate": [x, y]}
        """
        if not response:
            return None
        
        # Method 1: start_box='(x,y)' format
        match = re.search(r"start_box=['\"]?\((\d+),\s*(\d+)\)['\"]?", response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return [x, y]
        
        # Method 2: point='<point>x y</point>' format
        match = re.search(r"<point>(\d+)\s+(\d+)</point>", response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return [x, y]
        
        # Method 3: click(point='<point>x y</point>') format
        match = re.search(r"point=['\"]?<point>(\d+)\s+(\d+)</point>['\"]?", response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return [x, y]
        
        # Method 4: JSON format {"coordinate": [x, y]} (take only the first one)
        json_matches = re.findall(r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]', response)
        if json_matches:
            x, y = int(json_matches[0][0]), int(json_matches[0][1])
            return [x, y]
        
        # Method 5: Pure coordinates (x, y) or [x, y]
        match = re.search(r"[\(\[](\d+),\s*(\d+)[\)\]]", response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return [x, y]
        
        # Method 6: Space separated coordinates "x y"
        match = re.search(r"\b(\d{2,4})\s+(\d{2,4})\b", response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            # Verify coordinate validity (assuming image does not exceed 10000x10000)
            if 0 <= x <= 10000 and 0 <= y <= 10000:
                return [x, y]
        
        print(f"⚠️ UITars: Failed to parse coordinates from response: {response[:200]}")
        return None
