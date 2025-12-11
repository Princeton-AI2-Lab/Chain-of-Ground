import os
import json
import time
import re
import requests
from io import BytesIO
import base64

def convert_pil_image_to_base64(image, format="PNG"):
    b = BytesIO()
    image.save(b, format=format)
    return base64.b64encode(b.getvalue()).decode()

def round_by_factor(x, factor):
    return round(x / factor) * factor

def ceil_by_factor(x, factor):
    return ((x + factor - 1) // factor) * factor

def floor_by_factor(x, factor):
    return (x // factor) * factor

def smart_resize(height, width, min_pixels=100*28*28, max_pixels=16384*28*28, max_ratio=200, factor=28):
    oh, ow = height, width
    area = oh * ow
    if area < min_pixels:
        s = (min_pixels / area) ** 0.5
        nh = int(ceil_by_factor(oh * s, factor))
        nw = int(ceil_by_factor(ow * s, factor))
    elif area > max_pixels:
        s = (max_pixels / area) ** 0.5
        nh = int(floor_by_factor(oh * s, factor))
        nw = int(floor_by_factor(ow * s, factor))
    else:
        nh = int(round_by_factor(oh, factor))
        nw = int(round_by_factor(ow, factor))
    ratio = max(nh, nw) / min(nh, nw)
    if ratio > max_ratio:
        if nh > nw:
            nh = int(nw * max_ratio)
        else:
            nw = int(nh * max_ratio)
    nh = max(factor, nh)
    nw = max(factor, nw)
    return nh, nw

class UITarsBackend:
    def __init__(self, api_base="https://openrouter.ai/api/v1/chat/completions", model_name=None):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    def call(self, prompt, image, model_name=None, temperature=0.0, max_tokens=2048, max_retries=3):
        base64_image = convert_pil_image_to_base64(image, format="PNG")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name or self.model_name, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}], "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(max_retries):
            try:
                r = requests.post(self.api_base, headers=headers, json=payload, timeout=60)
                r.raise_for_status()
                j = r.json()
                return j['choices'][0]['message']['content']
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def parse_pixel_coordinates(self, response, image):
        if not response:
            return None
        try:
            response = re.sub(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\)', r'[\1, \2]', response)
            if '"coordinate"' in response:
                ob = response.count('[')
                cb = response.count(']')
                oc = response.count('{')
                cc = response.count('}')
                if ob > cb:
                    response += ']' * (ob - cb)
                if oc > cc:
                    response += '}' * (oc - cc)
            try:
                data = json.loads(response)
                coords = data.get("arguments", {}).get("coordinate", [])
                if len(coords) == 2:
                    return [float(coords[0]), float(coords[1])]
            except json.JSONDecodeError:
                pass
            m = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)', response)
            if m:
                return [float(m.group(1)), float(m.group(2))]
            m = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)
            if m:
                return [float(m.group(1)), float(m.group(2))]
        except Exception:
            pass
        return None
