import os  
import json  
import base64  
from io import BytesIO  
from PIL import Image  
import time  
import re 
from openai import OpenAI, BadRequestError  


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
        new_height = int(ceil_by_factor(orig_height * scale, factor))
        new_width = int(ceil_by_factor(orig_width * scale, factor))
    elif area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        new_height = int(floor_by_factor(orig_height * scale, factor))
        new_width = int(floor_by_factor(orig_width * scale, factor))
    else:
        new_height = int(round_by_factor(orig_height, factor))
        new_width = int(round_by_factor(orig_width, factor))
    
    ratio = max(new_height, new_width) / min(new_height, new_width)
    if ratio > max_ratio:
        if new_height > new_width:
            new_height = int(new_width * max_ratio)
        else:
            new_width = int(new_height * max_ratio)
    
    new_height = max(factor, new_height)
    new_width = max(factor, new_width)
    return new_height, new_width


def extract_first_bounding_box(text):
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
    return None

def extract_first_point(text):
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None

def extract_start_box_point(text):
    pattern_start_box = r"start_box='\((\d+),(\d+)\)'"
    pattern_point = r"point='\((\d+),(\d+)\)'"
    match = re.search(pattern_start_box, text, re.DOTALL)
    if not match:
        match = re.search(pattern_point, text, re.DOTALL)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None

def convert_pil_image_to_base64(image):  
    """Convert PIL Image to base64 string."""  
    buffered = BytesIO()  
    image.save(buffered, format="PNG")  
    img_str = base64.b64encode(buffered.getvalue()).decode()  
    return img_str  
  
  
class UITarsSingleMethod:  
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",  
                 api_base="https://openrouter.ai/api/v1",
                 http_referer=None,
                 x_title=None):  
        """
        Two-layer UI-TARS self-correction model: UI-TARS initial check + UI-TARS correction

        Args:
            uitars_model: UI-TARS model name
            api_base: OpenRouter API endpoint
            http_referer: Optional
            x_title: Optional
        """
        self.uitars_model = uitars_model  
        self.api_base = api_base   
          
    
        self.api_key = os.environ.get("OPENROUTER_API_KEY")  
        if not self.api_key:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable")
          
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )
        
        self.extra_headers = {}
        if http_referer:
            self.extra_headers["HTTP-Referer"] = http_referer
        if x_title:
            self.extra_headers["X-Title"] = x_title
          
        self.override_generation_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,  
        }  
          
        self.logs = []  
        self.debug_flag = True

      
    def load_model(self):
        """Load model"""
        pass
      
    def set_generation_config(self, **kwargs):
        """Set generation config"""
        # Disallow passing max_new_tokens; replace with max_tokens
        if "max_new_tokens" in kwargs:
            
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        
        supported_params = ["temperature", "top_p", "max_tokens", "stop", "n"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        self.override_generation_config.update(filtered_kwargs)
        self.debug_print(f"Updated generation config: {self.override_generation_config}")
      
    def debug_print(self, string):  
        self.logs.append(string)  
        if self.debug_flag:  
            print(string)  

    def call_uitars_api(self, prompt, image, is_allow_negative=False):
        """Call OpenRouter UI-TARS API"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        base64_image = convert_pil_image_to_base64(image)

     
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        try:
           
            response = self.client.chat.completions.create(
                model=self.uitars_model,
                messages=messages,
                extra_headers=self.extra_headers,
                stream=True,
                **self.override_generation_config  
            )

          
            response_text = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    response_text += content
                    print(content, end='', flush=True)
            self.debug_print(f"\nOpenRouter API response completed: {response_text}")
            return response_text

        except TypeError as e:
           
            if "unexpected keyword argument 'max_new_tokens'" in str(e):
                error_msg = "Error: OpenRouter API does not support max_new_tokens; use max_tokens instead!"
                self.debug_print(error_msg)
                raise ValueError(error_msg) from e
            else:
                self.debug_print(f"OpenRouter call failed: {e}")
                return None
        except BadRequestError as e:
            self.debug_print(f"OpenRouter call failed: {e}")
            return None
        except Exception as e:
            self.debug_print(f"OpenRouter call failed: {e}")
            return None

    def parse_pixel_coordinates_raw(self, response_text, image):
        """Parse pixel coordinates"""
        if not response_text:
            return None
        
        img_width, img_height = image.size
        click_point = None

        try:
            
            start_box_point = extract_start_box_point(response_text)
            if start_box_point:
                self.debug_print(f"Extracted start_box/point coordinates: {start_box_point}")
               
                new_height, new_width = smart_resize(img_height, img_width)
                self.debug_print(f"Image resolution adaptation: original ({img_width}x{img_height}) -> resized ({new_width}x{new_height})")
               
                x = int(start_box_point[0] / new_width * img_width)
                y = int(start_box_point[1] / new_height * img_height)
                click_point = [x, y]
        
          
            if not click_point:
                first_point = extract_first_point(response_text)
                if first_point:
                    self.debug_print(f"Extracted normalized point: {first_point}")
                
                    x = int(first_point[0] * img_width)
                    y = int(first_point[1] * img_height)
                    click_point = [x, y]
        
      
            if not click_point:
                bbox = extract_first_bounding_box(response_text)
                if bbox:
                    self.debug_print(f"Extracted bounding box: {bbox}")
              
                    x = int((bbox[0] + bbox[2]) / 2 * img_width)
                    y = int((bbox[1] + bbox[3]) / 2 * img_height)
                    click_point = [x, y]

     
            if click_point:
                click_point[0] = max(0, min(img_width, click_point[0]))
                click_point[1] = max(0, min(img_height, click_point[1]))
                self.debug_print(f"Final valid pixel coordinates: {click_point}")
                return click_point

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

    def ground_with_uitars_initial(self, instruction, image):
        """Layer 1: UI-TARS initial check"""
        width, height = image.size  
        
        prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...

## Action Space
click(point='<point>x1 y1</point>'')

## User Instruction
{instruction}"""  
    
     
        response = self.call_uitars_api(prompt, image)
    
        pixel_point = self.parse_pixel_coordinates_raw(response, image)  
    
        return pixel_point, response

    
    def ground_only_positive(self, instruction, image):
        """Main entry: single-layer UI-TARS"""
        self.logs = []    
        
        if isinstance(image, str):    
            image = Image.open(image).convert('RGB')    
        
        width, height = image.size    
        
        uitars_pixel_point, uitars_initial_response = self.ground_with_uitars_initial(instruction, image)    
        
        if uitars_pixel_point is None:    
            return {    
                "result": "negative",    
                "point": None,    
                "bbox": None,    
                "raw_response": {"uitars_initial": uitars_initial_response, "logs": self.logs}    
            }    
        
        # Normalize initial check result
        final_normalized_point = self.normalize_pixel_coordinates(uitars_pixel_point, width, height)    
        
        return {    
            "result": "positive" if final_normalized_point else "negative",    
            "point": final_normalized_point,    
            "bbox": None,    
            "raw_response": {    
                "uitars_initial": uitars_initial_response,    
                "logs": self.logs    
            }    
        }
      
    def ground_allow_negative(self, instruction, image):
        """Grounding with support for negative samples"""
        self.logs = []    
        
        if isinstance(image, str):    
            image = Image.open(image).convert('RGB')    
        
        width, height = image.size  

        # Prompt specialized for negative samples
        prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1.
If such element does not exist, output only the text 'Target not existent'.
The instruction is:
{instruction}"""

        # Call OpenRouter API (allow negative sample judgment)
        response = self.call_uitars_api(prompt, image, is_allow_negative=True)
        
        # Check negative sample indicator (case-insensitive)
        if response and "target not existent" in response.lower():
            self.debug_print("Negative sample detected: target element does not exist")
            return {
                "result": "negative",
                "point": None,
                "bbox": None,
                "raw_response": {"uitars_initial": response, "logs": self.logs}
            }

        # Parse coordinates
        pixel_point = self.parse_pixel_coordinates_raw(response, image)  
        if pixel_point is None:    
            return {    
                "result": "negative",    
                "point": None,    
                "bbox": None,    
                "raw_response": {"uitars_initial": response, "logs": self.logs}    
            }  

        # Normalize coordinates
        final_normalized_point = self.normalize_pixel_coordinates(pixel_point, width, height)    
        return {    
            "result": "positive" if final_normalized_point else "negative",    
            "point": final_normalized_point,    
            "bbox": None,    
            "raw_response": {    
                "uitars_initial": response,    
                "logs": self.logs    
            }    
        }
