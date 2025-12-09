            
import os
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import requests
import time
import re
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def convert_pil_image_to_base64(image, format="JPEG"):
    """Convert PIL Image to base64 string."""
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


class UITarsQwen3VLHybridMethod:  
    def __init__(self,
                 uitars_model="bytedance/ui-tars-1.5-7b",
                 qwen_refine_model="qwen3-vl-235b-a22b-instruct",  
                 qwen_final_model="qwen3-vl-32b-instruct",       
                 uitars_api_base="https://openrouter.ai/api/v1/chat/completions",
                 qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        Three-layer composite model: UITars initial check (OpenRouter) + Qwen refinement (Dashscope) + Qwen finalization (Dashscope)

        Args:
            uitars_model: UITars initial detection model name (OpenRouter format)
            qwen_refine_model: Qwen second-layer refinement model name (Dashscope format)
            qwen_final_model: Qwen third-layer final model name (Dashscope format)
            uitars_api_base: OpenRouter API endpoint for UITars
            qwen_api_base: Dashscope API endpoint for Qwen
        """
      
        self.uitars_model = uitars_model
        self.uitars_api_base = uitars_api_base
        self.uitars_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.uitars_api_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")

      
        self.qwen_refine_model = qwen_refine_model
        self.qwen_final_model = qwen_final_model
        self.qwen_api_base = qwen_api_base
        self.qwen_api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.qwen_api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")

        self.override_generation_config = {
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        self.logs = []
        self.debug_flag = True

    def plot_annotated_box(self, image, point, box_size=100, is_correct=None, alpha=80):
        """Annotate predicted point on image with semi-transparent box"""
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if point is not None:
            x, y = point

            if is_correct:
                color = (0, 0, 255, alpha)  # blue
            else:
                color = (255, 0, 0, alpha)  # red

            
            half_size = box_size // 2
            draw.rectangle(
                (x - half_size, y - half_size, x + half_size, y + half_size),
                fill=color,
                outline=None
            )

        annotated_image = image.convert('RGBA')
        annotated_image = Image.alpha_composite(annotated_image, overlay)
        return annotated_image.convert('RGB')

    def plot_annotated_circle(self, image, point, radius=15, is_correct=None, alpha=100):
        """Annotate predicted point on image with semi-transparent solid circle"""
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        if point is not None:
            x, y = point

            if is_correct:
                color = (0, 0, 255, alpha)  # blue
            else:
                color = (255, 0, 0, alpha)  # red

            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=None
            )

        annotated_image = image.convert('RGBA')
        annotated_image = Image.alpha_composite(annotated_image, overlay)
        return annotated_image.convert('RGB')

    def load_model(self):
        """Load model (API calls do not require actual loading)"""
        pass

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def debug_print(self, string):
        self.logs.append(string)
        if self.debug_flag:
            print(string)

    # ------------------------------ UITars Methods ------------------------------
    def call_openrouter_api(self, prompt, image, max_retries=3):
        """Call OpenRouter API (UITars)"""
        base64_image = convert_pil_image_to_base64(image, format="PNG")  

        headers = {
            "Authorization": f"Bearer {self.uitars_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.uitars_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": self.override_generation_config["temperature"],
            "max_tokens": self.override_generation_config["max_tokens"],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.uitars_api_base,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']

                self.debug_print(f"UITars API response status: {response.status_code}")
                self.debug_print(f"UITars full response:\n{content}")
                return content

            except Exception as e:
                self.debug_print(f"UITars API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def parse_uitars_pixel_coordinates(self, response, resized_width, resized_height):
        """Parse UITars pixel coordinates and adapt to Qwen coordinate system"""
        if not response:
            self.debug_print("Response is empty")
            return None

        try:
           
            response = re.sub(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\)', r'[\1, \2]', response)
            if '"coordinate"' in response:
                open_brackets = response.count('[')
                close_brackets = response.count(']')
                open_braces = response.count('{')
                close_braces = response.count('}')
                if open_brackets > close_brackets:
                    response += ']' * (open_brackets - close_brackets)
                if open_braces > close_braces:
                    response += '}' * (open_braces - close_braces)

           
            if "coordinate" in response:
                try:
                    data = json.loads(response)
                    coords = data.get("arguments", {}).get("coordinate", [])
                    if len(coords) == 2:
                        x_pixel = float(coords[0])
                        y_pixel = float(coords[1])
                        self.debug_print(f"UITars extracted pixel coordinates: [{x_pixel}, {y_pixel}]")
                        return [x_pixel, y_pixel]
                except json.JSONDecodeError:
                    pass

      
            match = re.search(r'"coordinate"\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)', response)
            if match:
                x_pixel = float(match.group(1))
                y_pixel = float(match.group(2))
                self.debug_print(f"UITars regex-extracted coordinates: [{x_pixel}, {y_pixel}]")
                return [x_pixel, y_pixel]

           
            match = re.search(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', response)
            if match:
                x_pixel = float(match.group(1))
                y_pixel = float(match.group(2))
                self.debug_print(f"UITars directly extracted coordinates: [{x_pixel}, {y_pixel}]")
                return [x_pixel, y_pixel]

        except Exception as e:
            self.debug_print(f"UITars coordinate parsing failed: {e}")

        return None

    def ground_with_uitars_initial(self, instruction, image):
        """Layer 1: UITars1.5-7b initial detection"""
        input_width, input_height = image.size  
    
        # Use smart_resize to compute adjusted dimensions
        resized_height, resized_width = smart_resize(  
            input_height,  
            input_width,  
            min_pixels=100*28*28,  
            max_pixels=16384*28*28,  
            max_ratio=200,  
            factor=28  
        )  
        resized_image = image.resize((resized_width, resized_height))  
    
        self.debug_print(f"Image resized: {input_width}x{input_height} -> {resized_width}x{resized_height}")
    
 
        prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.  
    
    ## Output Format  
    
    Action: ...  
    
    ## Action Space  
    click(point='<point>x1 y1</point>'')  
    
    ## User Instruction  
    {instruction}"""  
    
     
        response = self.call_openrouter_api(prompt, resized_image)  
    
      
        pixel_point = self.parse_uitars_pixel_coordinates_enhanced(response, resized_image)  
    
        return pixel_point, response, resized_image
    
    def parse_uitars_pixel_coordinates_enhanced(self, response_text, image):
        """Parse UITars returned pixel coordinates (enhanced)"""
        if not response_text:
            self.debug_print("Response is empty")
            return None  
        
        img_width, img_height = image.size  
        click_point = None  
    
        try:  
             
            start_box_point = self.extract_start_box_point(response_text)
            if start_box_point:  
                self.debug_print(f"Extracted start_box/point coordinates: {start_box_point}")
             
                new_height, new_width = smart_resize(img_height, img_width)  
                self.debug_print(f"Resolution adaptation: original({img_width}x{img_height}) -> adjusted({new_width}x{new_height})")
            
                x = int(start_box_point[0] / new_width * img_width)  
                y = int(start_box_point[1] / new_height * img_height)  
                click_point = [x, y]  
        
            if not click_point:  
                first_point = self.extract_first_point(response_text)
                if first_point:  
                    self.debug_print(f"Extracted normalized point coordinates: {first_point}")
                    x = int(first_point[0] * img_width)  
                    y = int(first_point[1] * img_height)  
                    click_point = [x, y]  
        
        
            if not click_point:  
                bbox = self.extract_first_bounding_box(response_text)
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
   
    def extract_start_box_point(self, text):  
        pattern_start_box = r"start_box='\((\d+),(\d+)\)'"  
        pattern_point = r"point='\((\d+),(\d+)\)'"  
        match = re.search(pattern_start_box, text, re.DOTALL)  
        if not match:  
            match = re.search(pattern_point, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2))]  
        return None  
    
    def extract_first_point(self, text):  
        pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
        match = re.search(pattern, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2))]  
        return None  
    
    def extract_first_bounding_box(self, text):  
        pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
        match = re.search(pattern, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]  
        return None

    # ------------------------------ Qwen Methods ------------------------------
    def call_dashscope_api(self, messages, model_name, max_retries=3):
        """Call Dashscope API (Qwen)"""
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
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
                    timeout=120  # increase timeout
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
        """Parse normalized coordinates [0-1000] from Dashscope response and convert to pixel coordinates"""
        if not response:
            self.debug_print("Response is empty")
            return None

        try:
       
            if '<tool_call>' not in response:
                self.debug_print(f"No <tool_call> tag found; raw response: {response[:200]}")
            
                try:
                    
                    coord_match = re.search(r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]', response)
                    if coord_match:
                        x_1000 = int(coord_match.group(1))
                        y_1000 = int(coord_match.group(2))
                        self.debug_print(f"Extracted coordinates via fallback logic: [{x_1000}, {y_1000}]")

                     
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
                self.debug_print(f"Fallback parse also failed: {fallback_e}")
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

    def refine_with_qwen_235b(self, instruction, annotated_image, initial_pixel_point, initial_response, is_correct=False):
        """Layer 2: Qwen3-VL-235B correction"""
        self.debug_print("=== Layer 2: Qwen3-VL-235B correction ===")
        resized_width, resized_height = annotated_image.size
        
        x_pixel, y_pixel = initial_pixel_point if initial_pixel_point else (resized_width//2, resized_height//2)
        
     
        x_1000 = int((x_pixel / resized_width) * 1000)
        y_1000 = int((y_pixel / resized_height) * 1000)
        
   
        prompt = f"""**CRITICAL: Search WITHIN the large red circle area.**  

Task: {instruction}  

The screenshot shows a LARGE RED CIRCLE at [{x_1000}, {y_1000}]. The target element should be INSIDE this red circle area.  

Your job:  
1. Focus on the area WITHIN the red circle  
2. Find the correct element for: "{instruction}"  
3. Output its center coordinates [x, y] where x,y ∈ [0,1000]  

The red circle is intentionally large to contain the target. Your coordinate should be inside this circle.  

Output format (STRICT):  
<tool_call>  
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
</tool_call>  

NO explanations. NO text outside tags."""
        
   
        base64_image = convert_pil_image_to_base64(annotated_image)
        system_message_content = [
            {"type": "text", "text": "You are a helpful assistant."},
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
        
        messages = [
            {"role": "system", "content": system_message_content},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        refined_response = self.call_dashscope_api(messages, model_name=self.qwen_refine_model)
        self.debug_print(f"Qwen235B correction response: {refined_response}")
        
        refined_pixel_point = self.parse_normalized_coordinates(refined_response, resized_width, resized_height)
        
        return refined_pixel_point, refined_response

    def refine_with_qwen_32b_final(self, instruction, image, first_pixel_point, second_pixel_point,
                                first_response, second_response):
        """Layer 3: Qwen3-VL-32B final correction"""
        self.debug_print("=== Layer 3: Qwen3-VL-32B final correction ===")
        resized_width, resized_height = image.size
        
        # Annotate both Layer 1 (red) and Layer 2 (blue) on original image
        annotated_image = image.copy()
        
        # Mark Layer 1 result (red semi-transparent circle)
        if first_pixel_point:
            annotated_image = self.plot_annotated_circle(
                annotated_image,
                first_pixel_point,
                radius=100,
                is_correct=False,  # red
                alpha=60
            )
        
        # Mark Layer 2 result (blue box)
        if second_pixel_point:    
            annotated_image = self.plot_annotated_box(    
                annotated_image,    
                second_pixel_point,    
                box_size=100,  # large box
                is_correct=True,  # blue
                alpha=60    
            )
        
        # Convert both layers' coordinates to 0–1000 normalized format
        first_x_1000 = int((first_pixel_point[0] / resized_width) * 1000) if first_pixel_point else 500
        first_y_1000 = int((first_pixel_point[1] / resized_height) * 1000) if first_pixel_point else 500
        second_x_1000 = int((second_pixel_point[0] / resized_width) * 1000) if second_pixel_point else 500
        second_y_1000 = int((second_pixel_point[1] / resized_height) * 1000) if second_pixel_point else 500
      
        system_message_content = [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": """    
    # Tools    
    
    You may call one or more functions to assist with the user query.    
    
    You are provided with function signatures within <tools></tools> XML tags:    
    <tools>    
    {"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* The screen's resolution is 1000x1000.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button.", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}}, "required": ["action", "coordinate"], "type": "object"}}}    
    </tools>    
    
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:    
    <tool_call>    
    {"name": <function-name>, "arguments": <args-json-object>}    
    </tool_call>"""}
        ]
        

        user_prompt = f"""You are a GUI automation assistant. The coordinate range is 0–1000 in both x and y.

Task: Precisely locate "{instruction}" within the blue square.

There are two reference areas in the image:
- Large red circle centered at [{first_x_1000}, {first_y_1000}] (broad area; target is inside it)
- Blue square centered at [{second_x_1000}, {second_y_1000}] (refined area; target is inside this square, and the square is fully within the red circle)

Focus only on the blue square. Ignore any color overlap.  
Find the exact center of "{instruction}" within the blue square and output the final coordinates.  
If unsure, choose the most likely point inside the blue area.

Available tool: computer_use(action: str, coordinate: list[int, int]) with action="left_click" and coordinate=[x, y] where x and y are integers in [0,1000].

Output only this JSON:
<tool_call> 
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}
</tool_call>

Return ONLY the JSON, no explanation."""
        
    
        base64_image = convert_pil_image_to_base64(annotated_image)
        messages = [
            {"role": "system", "content": system_message_content},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
   
        response = self.call_dashscope_api(messages, model_name=self.qwen_final_model)
        self.debug_print(f"Qwen32B final correction response: {response}")
        

        final_pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)
        
        return final_pixel_point, response

    def ground_only_positive(self, instruction, image):
        """Main entry: three-layer processing flow"""
        self.logs = []
        
        # Load original image
        if isinstance(image, str):
            image_path = image
            image = Image.open(image_path).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a file path or PIL Image")
            
        original_width, original_height = image.size
        
        # Layer 1: UITars1.5-7b initial detection
        uitars_pixel_point, uitars_response, resized_image = self.ground_with_uitars_initial(instruction, image)
        resized_width, resized_height = resized_image.size
        
        if not uitars_pixel_point:
            return {
                "result": "negative",
                "point": None,
                "bbox": None,
                "raw_response": {"uitars_initial": uitars_response, "logs": self.logs}
            }
        
        # Save Layer 1 annotated image - use large red circle
        annotated_layer1 = self.plot_annotated_circle(
            resized_image,
            uitars_pixel_point,
            radius=100,  
            is_correct=False,
            alpha=60  
        )
        annotated_layer1.save("layer1_large_red_circle.png")
        self.debug_print("Layer 1 completed: UITars initial check (large red circle annotation)")
        
        time.sleep(1)  # Avoid API rate limits
    
        # Layer 2: Qwen3-VL-235B correction - search within the red circle
        qwen_refine_point, qwen_refine_response = self.refine_with_qwen_235b(
            instruction, annotated_layer1, uitars_pixel_point, uitars_response, is_correct=False
        )
        
        # Fallback on correction failure
        if qwen_refine_point is None:
            self.debug_print("Qwen235B correction failed; fallback to UITars initial result")
            qwen_refine_point = uitars_pixel_point
        
        # Save Layer 2 annotated image - overlay blue box on Layer 1
        annotated_layer2 = annotated_layer1.copy()  # retain red circle
        annotated_layer2 = self.plot_annotated_box(
            annotated_layer2,
            qwen_refine_point,
            box_size=100,  
            is_correct=True,  
            alpha=60
        )
        annotated_layer2.save("layer2_blue_box_in_red_circle.png")
        self.debug_print("Layer 2 completed: Qwen235B correction (blue box annotation)")

        # Layer 3: Qwen3-VL-32B final correction - precise localization within the blue box
        final_pixel_point, qwen_final_response = self.refine_with_qwen_32b_final(
            instruction, resized_image, uitars_pixel_point, qwen_refine_point,
            uitars_response, qwen_refine_response
        )
        
        # Fallback on Layer 3 failure
        if final_pixel_point is None:
            self.debug_print("Layer 3 correction failed; fallback to Layer 2 result")
            final_pixel_point = qwen_refine_point
        
        # Save Layer 3 annotated image - show all three-layer results
        annotated_layer3 = annotated_layer2.copy()  
        annotated_layer3 = self.plot_annotated_circle(
            annotated_layer3,
            final_pixel_point,
            radius=10,  
            is_correct=None,  
            alpha=200
        )
        annotated_layer3.save("layer3_final_precise_point.png")
        self.debug_print("Layer 3 completed: Qwen32B finalization (small green circle annotation)")
        
        # Map coordinates back to original image
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        final_original_point = [
            final_pixel_point[0] * scale_x,
            final_pixel_point[1] * scale_y
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
                "uitars_initial": uitars_response,
                "qwen_refine_235b": qwen_refine_response,
                "qwen_final_32b": qwen_final_response,
                "logs": self.logs
            }
        }
    
    def ground_allow_negative(self, instruction, image):
        """Support negative sample grounding"""
        return self.ground_only_positive(instruction, image)
