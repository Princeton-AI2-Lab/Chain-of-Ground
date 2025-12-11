import os    
import json    
import base64    
from io import BytesIO    
from PIL import Image, ImageDraw    
import requests    
import time    
import re 
from openai import OpenAI    
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize  


def convert_pil_image_to_base64(image):    
    """Convert PIL Image to base64 string."""    
    buffered = BytesIO()    
    image.save(buffered, format="JPEG") 
    img_str = base64.b64encode(buffered.getvalue()).decode()    
    return img_str


def round_by_factor(x, factor):  
    return round(x / factor) * factor  
  
def ceil_by_factor(x, factor):  
    return ((x + factor - 1) // factor) * factor  
  
def floor_by_factor(x, factor):  
    return (x // factor) * factor  
  
def smart_resize_uitars(height, width, min_pixels=100*28*28, max_pixels=16384*28*28, max_ratio=200, factor=28):
    """UITars-specific smart_resize - differs from transformers version"""
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
  

def extract_start_box_point(text):  
    pattern_start_box = r"start_box='\((\d+),(\d+)\)'"  
    pattern_point = r"point='\((\d+),(\d+)\)'"  
    match = re.search(pattern_start_box, text, re.DOTALL)  
    if not match:  
        match = re.search(pattern_point, text, re.DOTALL)  
    if match:  
        return [float(match.group(1)), float(match.group(2))]  
    return None  
  
def extract_first_point(text):  
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
    match = re.search(pattern, text, re.DOTALL)  
    if match:  
        return [float(match.group(1)), float(match.group(2))]  
    return None  
  
def extract_point_tag(text):
    pattern = r"<point>\s*(\d+)\s+(\d+)\s*</point>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [int(match.group(1)), int(match.group(2))]
    return None

def extract_first_bounding_box(text):  
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
    match = re.search(pattern, text, re.DOTALL)  
    if match:  
        return [float(match.group(1)), float(match.group(2)),   
                float(match.group(3)), float(match.group(4))]  
        return None    


class Qwen3VL32BTripleMethod:  
    def __init__(self,              
                    qwen_32b_model="qwen3-vl-32b-instruct",  
                    uitars_model="bytedance/ui-tars-1.5-7b", 
                    qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",  
                    uitars_api_base="https://openrouter.ai/api/v1"): 
        """Three-layer composite: Qwen-32B initial detection + UITars refinement + Qwen-32B final verification"""
        
        self.qwen_32b_model = qwen_32b_model  
        self.uitars_model = uitars_model   
        self.qwen_api_base = qwen_api_base  
        self.uitars_api_base = uitars_api_base  
        
        
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")  
        if not self.dashscope_api_key:
            raise ValueError("Please set DASHSCOPE_API_KEY environment variable")
        
         
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")  
        if not self.openrouter_api_key:
            raise ValueError("Please set OPENROUTER_API_KEY environment variable")
         

        self.uitars_client = OpenAI(  
            base_url=self.uitars_api_base,  
            api_key=self.openrouter_api_key  
        )  
        
        self.override_generation_config = {  
            "temperature": 0.0,  
            "max_tokens": 2048,  
        }  
        
        self.logs = []  
        self.debug_flag = True
    
    def call_uitars_api(self, prompt, image, max_retries=3):
        """Call OpenRouter UITars API"""
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        
      
        buffered = BytesIO()  
        image.save(buffered, format="PNG")  
        base64_image = base64.b64encode(buffered.getvalue()).decode()  
        
        messages = [  
            {  
                "role": "system",  
                "content": [  
                    {"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}  
                ]  
            },  
            {  
                "role": "user",  
                "content": [  
                    {  
                        "type": "image_url",  
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}  
                    },  
                    {"type": "text", "text": prompt}  
                ]  
            }  
        ]  
        
        for attempt in range(max_retries):
            try:
                response = self.uitars_client.chat.completions.create(
                    model=self.uitars_model,
                    messages=messages,
                    stream=True,
                    temperature=0.0,
                    max_tokens=2048
                )

                response_text = ""
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        response_text += content

                self.debug_print(f"UITars API response: {response_text}")
                return response_text
            except Exception as e:
                self.debug_print(f"UITars API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None
    
    def parse_uitars_pixel_coordinates(self, response_text, image):
        """Parse UITars returned pixel coordinates"""
        if not response_text:  
            return None  
        
        img_width, img_height = image.size  
        click_point = None  
        
        try:  
            
            start_box_point = extract_start_box_point(response_text)
            if start_box_point:  
                self.debug_print(f"Extracted start_box/point coordinates: {start_box_point}")
                 
                new_height, new_width = smart_resize_uitars(img_height, img_width)  
                self.debug_print(f"UITars internal size: {new_width}x{new_height}, actual image: {img_width}x{img_height}")
               
                x = int(start_box_point[0] * img_width / new_width)  
                y = int(start_box_point[1] * img_height / new_height)  
                click_point = [x, y]  
            
           
            if not click_point:  
                first_point = extract_first_point(response_text)
                if first_point:  
                    self.debug_print(f"Extracted normalized point coordinates: {first_point}")
                    x = int(first_point[0] * img_width)  
                    y = int(first_point[1] * img_height)  
                    click_point = [x, y]  
            
         
            if not click_point:
                explicit_point = extract_point_tag(response_text)
                if explicit_point:
                    self.debug_print(f"Extracted explicit point tag: {explicit_point}")
                    x = int(explicit_point[0])
                    y = int(explicit_point[1])
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


    def plot_annotated_circle(self, image, point, radius=15, is_correct=None, alpha=100):
        """Annotate predicted point on image"""
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))    
        draw = ImageDraw.Draw(overlay)    
          
        if point is not None:    
            x, y = point    
              
            if is_correct:
                color = (0, 191, 255, alpha)  # blue
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

    def plot_annotated_box(self, image, point, box_size=100, is_correct=None, alpha=80):
        """Annotate predicted point on image"""
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
        
    def load_model(self):
        """Load model (API calls do not require actual loading)"""
        pass    
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)  


    def call_dashscope_api(self, messages, model_name=None, max_retries=3):
        """Call Dashscope API (Qwen)"""
        headers = {  
            "Authorization": f"Bearer {self.dashscope_api_key}",  
            "Content-Type": "application/json"  
        }  
        
        payload = {
                "model": model_name or self.qwen_32b_model,
                "messages": messages,
                "temperature": self.override_generation_config.get("temperature", 0.0),
                "max_tokens": self.override_generation_config.get("max_tokens", 4096),
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


    def ground_with_qwen_32b_initial(self, instruction, image):
        """Layer 1: Qwen3-VL-32B initial detection"""
        self.debug_print("=== Layer 1: Qwen3-VL-32B initial detection ===")
        
        input_width, input_height = image.size  
        
        resized_height, resized_width = smart_resize(  
            input_height,  
            input_width,  
            factor=32,  
            min_pixels=3136,  
            max_pixels=2007040   
        )  
        resized_image = image.resize((resized_width, resized_height))  
        
        self.debug_print(f"Image resized: {input_width}x{input_height} -> {resized_width}x{resized_height}")
        
   
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
        
     
        user_prompt = f"""**CRITICAL: You MUST output in the exact format specified. Do NOT provide explanations.**  
    
    Task: {instruction}  
    
    Output ONLY:  
    <tool_call>  
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
    </tool_call>  
    
    Where x,y are integers in [0, 1000] range."""  
        
         
        base64_image = convert_pil_image_to_base64(resized_image)  
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
        
        
        response = self.call_dashscope_api(messages, model_name=self.qwen_32b_model)  
        self.debug_print(f"Qwen-32B initial detection response: {response}")
        
        
        pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)  
        
        return pixel_point, response, resized_image


    def refine_with_uitars(self, instruction, image, initial_pixel_point, initial_response):
        """Layer 2: UITars refinement"""
        self.debug_print("=== Layer 2: UITars refinement ===")
        
        resized_width, resized_height = image.size    
        
     
        if initial_pixel_point:    
            annotated_image = self.plot_annotated_circle(    
                image,    
                initial_pixel_point,    
                radius=100,    
                is_correct=False,    
                alpha=60    
            )    
        else:    
            annotated_image = image    
        
        prompt = f"""**CRITICAL: You MUST search within the red circle area.**  
    
    Task: {instruction}  
    
    The screenshot shows a large red semi-transparent circle marking the initial detection area.  
    
    **MANDATORY REQUIREMENT:**  
    - The target "{instruction}" is GUARANTEED to be inside the red circle  
    - You MUST output coordinates that fall within the red circle boundary  
    - DO NOT search outside the red circle  
    
    **Your task:**  
    1. Carefully examine ALL UI elements inside the red circle  
    2. Identify the element that matches: "{instruction}"  
    3. Output the center coordinates of that element  
    
    ## Output Format  
    
    Action: click(point='<point>x1 y1</point>')  
    
    Where x1, y1 are pixel coordinates within the red circle area."""    
          
        response = self.call_uitars_api(prompt, annotated_image)    
         
        refined_pixel_point = self.parse_uitars_pixel_coordinates(response, annotated_image)    
        
        return refined_pixel_point, response
    
    def refine_with_qwen_32b(self, instruction, image, first_pixel_point, second_pixel_point,
                     first_response, second_response):
        """Layer 3: Qwen3-VL-32B final evaluation and correction"""
        self.debug_print("=== Layer 3: Qwen3-VL-32B final evaluation ===")
        resized_width, resized_height = image.size      
            
        annotated_image = image.copy()      
            
        if first_pixel_point:      
            annotated_image = self.plot_annotated_circle(      
                annotated_image,      
                first_pixel_point,      
                radius=100,    
                is_correct=False,      
                alpha=60      
            )      
            
        if second_pixel_point:      
            annotated_image = self.plot_annotated_box(      
                annotated_image,      
                second_pixel_point,      
                box_size=100,    
                is_correct=True,      
                alpha=60      
            )      
        
        
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
        

        user_prompt = f"""You are a GUI automation assistant performing final verification. The coordinate range is 0–1000 in both x and y.  
    
    Task: Locate "{instruction}"  
    
    **Previous Detection Results:**  
    - Layer 1 (Red circle): Initial detection at [{first_x_1000}, {first_y_1000}] with radius ~100  
    - Layer 2 (Blue square): Refined detection at [{second_x_1000}, {second_y_1000}] (should be inside red circle)  
    
    **Your Evaluation Process:**  
    
    1. **FIRST: Check if blue square is inside red circle**  
    - If YES: The blue square is the refined search area. Find "{instruction}" precisely within the blue square.  
    - If NO: This indicates the target may NOT be in the red circle. You must independently search the ENTIRE screenshot.  
    
    2. **Decision Logic:**  
    - Blue square inside red circle → Trust the refinement, output coordinates within blue square  
    - Blue square outside red circle → Ignore both markers, independently find "{instruction}" anywhere in the screenshot  
    
    3. **Output the final coordinates**  
    
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
        
      
        response = self.call_dashscope_api(messages, model_name=self.qwen_32b_model)
        self.debug_print(f"Qwen-32B final evaluation response: {response}")
        
           
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
        
        # Layer 1: Qwen3-VL-32B initial detection
        qwen_32b_pixel_point, qwen_32b_initial_response, resized_image = self.ground_with_qwen_32b_initial(instruction, image)  
        resized_width, resized_height = resized_image.size  
        
        if not qwen_32b_pixel_point:  
            return {  
                "result": "negative",  
                "point": None,  
                "bbox": None,  
                "raw_response": {"qwen_32b_initial": qwen_32b_initial_response, "logs": self.logs}  
            }  
        
        # Save Layer 1 annotated image
        annotated_layer1 = self.plot_annotated_circle(  
            resized_image,  
            qwen_32b_pixel_point,  
            radius=100,  
            is_correct=False,  
            alpha=60  
        )  
        annotated_layer1.save("layer1_large_red_circle.png")  
        self.debug_print("Layer 1 completed: large red circle annotation")
        
        time.sleep(1)  
        
        # Layer 2: UITars refinement - search within red circle
        uitars_refined_pixel_point, uitars_refined_response = self.refine_with_uitars(  
            instruction, annotated_layer1, qwen_32b_pixel_point, qwen_32b_initial_response  
        )  
        
        if uitars_refined_pixel_point is None:  
            self.debug_print("Layer 2 correction failed; fallback to Layer 1 result")
            uitars_refined_pixel_point = qwen_32b_pixel_point  
        
        # Save Layer 2 annotated image
        annotated_layer2 = annotated_layer1.copy()
        annotated_layer2 = self.plot_annotated_box(  
            annotated_layer2,  
            uitars_refined_pixel_point,  
            box_size=100,  
            is_correct=True,  
            alpha=60  
        )  
        annotated_layer2.save("layer2_blue_box.png")  
        self.debug_print("Layer 2 completed: blue box annotation (inside red circle)")
          
        time.sleep(1)  
          
        # Layer 3: Qwen-32B finalization
        qwen_32b_final_pixel_point, qwen_32b_final_response = self.refine_with_qwen_32b(    
            instruction,                 
            annotated_layer2,             
            qwen_32b_pixel_point,          
            uitars_refined_pixel_point,   
            qwen_32b_initial_response,     
            uitars_refined_response      
        )    
            
        if qwen_32b_final_pixel_point is None:    
            self.debug_print("Layer 3 correction failed; fallback to Layer 2 result")
            final_pixel_point = uitars_refined_pixel_point    
        else:    
            self.debug_print(f"Layer 3 corrected coordinates: {qwen_32b_final_pixel_point}")
            final_pixel_point = qwen_32b_final_pixel_point  
          
        # Save Layer 3 annotated image
        annotated_layer3 = annotated_layer2.copy()  
        annotated_layer3 = self.plot_annotated_circle(  
            annotated_layer3,  
            final_pixel_point,  
            radius=10,  
            is_correct=None,  
            alpha=200  
        )  
        annotated_layer3.save("layer3_final_precise_point.png")  
        self.debug_print("Layer 3 completed: final precise localization")
          
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
                "qwen_32b_initial": qwen_32b_initial_response,  
                "uitars_refined": uitars_refined_response,  
                "qwen_32b_final": qwen_32b_final_response,  
                "logs": self.logs  
            }  
        }  
      
    def ground_allow_negative(self, instruction, image):
        """Support negative sample grounding"""
        return self.ground_only_positive(instruction, image)  
  
