import os
import re
import json
import time
from openai import OpenAI

class QwenBackend:
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
    
    def call(self, messages, model_name=None, temperature=0.0, max_tokens=4096, max_retries=5):
        """Call Qwen model (OpenRouter)"""
        model = model_name or self.model_name
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                
                # Check for completely empty response
                if not content or content.strip() == "":
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"⚠️ Qwen API returned empty response after {max_retries} attempts")
                        return None
                
                # Check for empty <tool_call> tag
                if content.strip() in ["<tool_call>", "<tool_call>\n", "</tool_call>", "<tool_call></tool_call>"]:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"⚠️ Qwen API returned empty <tool_call> after {max_retries} attempts")
                        return None
                
                # Check for incomplete response
                if content.strip().startswith("<tool_call>") and "</tool_call>" not in content:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Qwen API call failed after {max_retries} attempts: {e}")
                    return None
                print(f"⚠️ Qwen API attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)
        
        return None
    
    def parse_normalized_coordinates(self, response, image_width, image_height):
        """
        Parse normalized coordinates [0-1000] and convert to pixel coordinates
        """
        if not response or not response.strip():
            return None
        
        try:
            # Method 1: Standard <tool_call> format
            if '<tool_call>' in response and '</tool_call>' in response:
                tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
                if tool_match:
                    tool_json = tool_match.group(1).strip()
                    if not tool_json:
                        return None
                    
                    try:
                        data = json.loads(tool_json)
                        coords = data.get("arguments", {}).get("coordinate", [])
                        
                        if len(coords) == 2:
                            x_norm = float(coords[0])
                            y_norm = float(coords[1])
                            
                            # Convert [0-1000] -> pixel coordinates
                            x_pixel = (x_norm / 1000.0) * image_width
                            y_pixel = (y_norm / 1000.0) * image_height
                            return [x_pixel, y_pixel]
                    except json.JSONDecodeError:
                        pass
            
            # Method 2: Fallback - Extract "coordinate": [x, y] directly
            coord_match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)
            if coord_match:
                x_norm = float(coord_match.group(1))
                y_norm = float(coord_match.group(2))
                
                x_pixel = (x_norm / 1000.0) * image_width
                y_pixel = (y_norm / 1000.0) * image_height
                return [x_pixel, y_pixel]
            
            # Method 3: Fallback - Pure array [x, y]
            array_match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)
            if array_match:
                x_norm = float(array_match.group(1))
                y_norm = float(array_match.group(2))
                
                # Determine if it is normalized coordinates or pixel coordinates
                if x_norm <= 1000 and y_norm <= 1000:
                    x_pixel = (x_norm / 1000.0) * image_width
                    y_pixel = (y_norm / 1000.0) * image_height
                    return [x_pixel, y_pixel]
                else:
                    return [x_norm, y_norm]
            
            # Method 4: Fallback - Parentheses format (x, y)
            paren_match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', response)
            if paren_match:
                x_norm = float(paren_match.group(1))
                y_norm = float(paren_match.group(2))
                if x_norm <= 1000 and y_norm <= 1000:
                    x_pixel = (x_norm / 1000.0) * image_width
                    y_pixel = (y_norm / 1000.0) * image_height
                    return [x_pixel, y_pixel]
            
            # Method 5: Fallback - Space separated
            space_match = re.search(r'\b(\d{1,4}\.?\d*)\s+(\d{1,4}\.?\d*)\b', response)
            if space_match:
                x_norm = float(space_match.group(1))
                y_norm = float(space_match.group(2))
                if x_norm <= 1000 and y_norm <= 1000:
                    x_pixel = (x_norm / 1000.0) * image_width
                    y_pixel = (y_norm / 1000.0) * image_height
                    return [x_pixel, y_pixel]
            
            print(f"⚠️ Qwen: Failed to parse coordinates from response: {response[:200]}")
            return None
            
        except Exception as e:
            print(f"⚠️ Qwen coordinate parsing exception: {e}")
            return None
    
    def parse_pixel_coordinates(self, response, image):
        """Parse pixel coordinates (for some special scenarios)"""
        if not response:
            return None
        
        iw, ih = image.size
        return self.parse_normalized_coordinates(response, iw, ih)
