import os
import json
import time
import re
import requests

class QwenBackend:
    def __init__(self, api_base="https://openrouter.ai/api/v1/chat/completions", model_name=None):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    def call(self, messages, model_name=None, temperature=0.0, max_tokens=4096, max_retries=3):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name or self.model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        for attempt in range(max_retries):
            try:
                r = requests.post(self.api_base, headers=headers, json=payload, timeout=120)
                r.raise_for_status()
                j = r.json()
                return j['choices'][0]['message']['content']
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def parse_normalized_coordinates(self, response, resized_width, resized_height):
        if not response:
            return None
        try:
            if '<tool_call>' not in response:
                try:
                    m = re.search(r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]', response)
                    if m:
                        x = int(m.group(1))
                        y = int(m.group(2))
                        return [(x/1000.0)*resized_width, (y/1000.0)*resized_height]
                    return None
                except Exception:
                    return None
            t = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
            if not t:
                return None
            data = json.loads(t.group(1).strip())
            coords = data.get("arguments", {}).get("coordinate", [])
            if len(coords) == 2:
                x = float(coords[0])
                y = float(coords[1])
                return [(x/1000.0)*resized_width, (y/1000.0)*resized_height]
            return None
        except json.JSONDecodeError:
            try:
                m = re.search(r'\[(\d+),\s*(\d+)\]', response)
                if m:
                    x = int(m.group(1))
                    y = int(m.group(2))
                    return [(x/1000.0)*resized_width, (y/1000.0)*resized_height]
            except Exception:
                pass
        except Exception:
            pass
        return None
