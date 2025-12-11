import os    
import json    
import base64    
from io import BytesIO    
from PIL import Image, ImageDraw    
import requests    
import time    
import re   
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize  


def convert_pil_image_to_base64(image):    
    """Convert PIL Image to base64 string."""    
    buffered = BytesIO()    
    image.save(buffered, format="JPEG")   
    img_str = base64.b64encode(buffered.getvalue()).decode()    
    return img_str    


class Qwen3VL235BMethod:  
    def __init__(self,   
                
                 qwen_model="qwen3-vl-235b-a22b-instruct",  
                 qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"):    
        """
        Two-layer composite model: Qwen initial check (Dashscope) + Qwen correction (Dashscope)

        Args:
            uitars_model: Keep original parameter (backward compatibility)
            qwen_model: Qwen model name (Dashscope format)
            uitars_api_base: Keep original parameter (backward compatibility)
            qwen_api_base: Dashscope API endpoint for Qwen
        """

        self.qwen_model = qwen_model    
        self.qwen_api_base = qwen_api_base   
          
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")  
        if not self.dashscope_api_key:
            raise ValueError("Please set the DASHSCOPE_API_KEY environment variable")
            
        self.override_generation_config = {    
            "temperature": 0.0,    
            "max_tokens": 2048,    
        }    
            
        self.logs = []    
        self.debug_flag = True  
      
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)  


    def call_dashscope_api(self, messages, max_retries=3):
        """Call Dashscope API"""
        headers = {  
            "Authorization": f"Bearer {self.dashscope_api_key}",  
            "Content-Type": "application/json"  
        }  
        
        payload = {  
            "model": self.qwen_model,  
            "messages": messages,  
            "temperature": 0.0,  
            "max_tokens": 4096,    
        }  
        
        for attempt in range(max_retries):  
            try:  
                response = requests.post(  
                    self.qwen_api_base + "/chat/completions",  
                    headers=headers,  
                    json=payload,  
                    timeout=120   
                )  
                response.raise_for_status()  
                result = response.json()  
                content = result['choices'][0]['message']['content']  
                
             
                if content.strip() == "<tool_call>" or content.strip() == "<tool_call>\n":
                    self.debug_print(f"Empty <tool_call> tag detected (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:  
                        time.sleep(2 ** attempt)  
                        continue  
                
                self.debug_print(f"Qwen API response status: {response.status_code}")
                self.debug_print(f"Qwen full response:\n{content}")
                return content  
                
            except Exception as e:
                self.debug_print(f"Qwen API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:  
                    time.sleep(2 ** attempt)  
        
        return None  


    def parse_normalized_coordinates(self, response, resized_width, resized_height):
        """Parse Dashscope normalized coordinates [0–1000] and convert to pixel coordinates"""
        if not response:
            self.debug_print("Empty response")
            return None    
            
        try:    
           
            if '<tool_call>' not in response:
                self.debug_print(f"No <tool_call> tag found; raw response: {response[:200]}")
                
                try:  
                   
                    coord_match = re.search(r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]', response)  
                    if coord_match:  
                        x_1000 = int(coord_match.group(1))  
                        y_1000 = int(coord_match.group(2))  
                        self.debug_print(f"Coordinates extracted via fallback: [{x_1000}, {y_1000}]")
                        
                    
                        x_pixel = (x_1000 / 1000.0) * resized_width    
                        y_pixel = (y_1000 / 1000.0) * resized_height    
                        self.debug_print(f"Normalized coordinates: [{x_1000}, {y_1000}] -> Pixel coordinates: [{x_pixel:.1f}, {y_pixel:.1f}]")
                        return [x_pixel, y_pixel]  
                    else:  
                        self.debug_print("Fallback failed to extract coordinates; returning None")
                        return None  
                except Exception as e:
                    self.debug_print(f"Fallback logic failed: {e}")
                    return None  
            
         
            tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)  
            if not tool_match:
                self.debug_print("Found <tool_call> tag but could not extract content")
                return None  
                
            tool_json = tool_match.group(1).strip()  
            data = json.loads(tool_json)  
            coords = data.get("arguments", {}).get("coordinate", [])  
                
            if len(coords) == 2:    
                 
                x_normalized = float(coords[0])    
                y_normalized = float(coords[1])    
                    
               
                x_pixel = (x_normalized / 1000.0) * resized_width    
                y_pixel = (y_normalized / 1000.0) * resized_height    
                    
                self.debug_print(f"Normalized coordinates: [{x_normalized}, {y_normalized}] -> Pixel coordinates: [{x_pixel:.1f}, {y_pixel:.1f}]")
                return [x_pixel, y_pixel]    
                
            self.debug_print("Invalid coordinate format")
            return None    
                
        except json.JSONDecodeError as e:
            self.debug_print(f"JSON parse failed: {e}")
       
            try:  
                coord_match = re.search(r'\[(\d+),\s*(\d+)\]', response)  
                if coord_match:  
                    x_1000 = int(coord_match.group(1))  
                    y_1000 = int(coord_match.group(2))  
                    self.debug_print(f"JSON parse failed; extracted via regex: [{x_1000}, {y_1000}]")
                    x_pixel = (x_1000 / 1000.0) * resized_width    
                    y_pixel = (y_1000 / 1000.0) * resized_height  
                    return [x_pixel, y_pixel]  
            except Exception as fallback_e:
                self.debug_print(f"Fallback parsing failed: {fallback_e}")
        except Exception as e:
            self.debug_print(f"Coordinate parsing failed: {e}")
            
        return None 


    def normalize_pixel_coordinates(self, pixel_point, width, height):
        """Normalize pixel coordinates to [0,1] range"""
        if pixel_point is None:    
            return None    
            
        try:    
            x_pixel, y_pixel = pixel_point    
            x_norm = max(0.0, min(1.0, x_pixel / width))    
            y_norm = max(0.0, min(1.0, y_pixel / height))    
            self.debug_print(f"Normalized: [{x_pixel}, {y_pixel}] -> [{x_norm:.4f}, {y_norm:.4f}]")
            return [x_norm, y_norm]    
        except Exception as e:
            self.debug_print(f"Normalization failed: {e}")
            return None


    def ground_with_qwen_initial(self, instruction, image):
        """Layer 1: Qwen3-VL initial check"""
        input_width, input_height = image.size    
        
      
        resized_height, resized_width = smart_resize(    
            input_height,    
            input_width,    
            factor=32,    
            min_pixels=3136,    
            max_pixels=2007040    
        )    
        resized_image = image.resize((resized_width, resized_height))    
        
        self.debug_print(f"Image scale: {input_width}x{input_height} -> {resized_width}x{resized_height}")
        
     
        system_message_content = [    
            {    
                "type": "text",    
                "text": "You are a helpful assistant."    
            },    
            {    
                "type": "text",    
                "text": """    
    
    # Tools    
    
    You may call one or more functions to assist with the user query.    
    
    You are provided with function signatures within <tools></tools> XML tags:    
    <tools>    
    {"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* The screen's resolution is 1000x1000.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button.", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}}, "required": ["action", "coordinate"], "type": "object"}}}    
    </tools>    
    
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:    
    <tool_call>    
    {"name": <function-name>, "arguments": <args-json-object>}    
    </tool_call>"""    
            }    
        ]    
    
 
        user_prompt = f"""**CRITICAL: You MUST output in the exact format specified. Do NOT provide explanations.**  
    
    Task: {instruction}  
    
    Output ONLY:  
    <tool_call>  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    </tool_call>  
    
    Where x,y are integers in [0, 1000] range."""  
    
    
        base64_image = convert_pil_image_to_base64(resized_image)    
        messages = [    
            {    
                "role": "system",    
                "content": system_message_content    
            },    
            {    
                "role": "user",    
                "content": [    
                    {    
                        "type": "image_url",    
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}    
                    },    
                    {"type": "text", "text": user_prompt}    
                ]    
            }    
        ]    
    
        response = self.call_dashscope_api(messages)    
        
      
        pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)    
        
        return pixel_point, response, resized_image  


    def ground_only_positive(self, instruction, image):
        """Main entry"""
        self.logs = []    

        if isinstance(image, str):    
            image_path = image    
            image = Image.open(image_path).convert('RGB')    
        elif not isinstance(image, Image.Image):    
            raise ValueError("image must be a file path or PIL Image")    
            
        original_width, original_height = image.size    
        
        # Layer 1: Qwen3-VL initial check
        qwen_pixel_point, qwen_response, resized_image = self.ground_with_qwen_initial(instruction, image)    
        resized_width, resized_height = resized_image.size    
        
       
        # Map coordinates back to original image
        scale_x = original_width / resized_width    
        scale_y = original_height / resized_height    
        final_original_point = [    
            qwen_pixel_point[0] * scale_x,    
            qwen_pixel_point[1] * scale_y    
        ]    
        
        # Normalize to [0,1]
        final_normalized_point = self.normalize_pixel_coordinates(    
            final_original_point,     
            original_width,     
            original_height    
        )    
        
        return {    
            "result": "positive" if final_normalized_point else "negative",    
            "point": final_normalized_point,    
            "bbox": None,    
            "raw_response": {    
                "qwen_initial": qwen_response,      
                "logs": self.logs    
            }    
        }
    
    def ground_allow_negative(self, instruction, image):
        """Grounding with support for negative samples"""
        return self.ground_only_positive(instruction, image)  
