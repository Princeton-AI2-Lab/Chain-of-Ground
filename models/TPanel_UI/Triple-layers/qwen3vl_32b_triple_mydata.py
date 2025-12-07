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


class Qwen3VL32BTripleMethod:  
    def __init__(self,             
         qwen_32b_model="qwen3-vl-32b-instruct",    
         qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"):    
        """
        Three-layer Qwen-32B composite model: Qwen-32B initial check + Qwen-32B correction + Qwen-32B final validation

        Note: All three layers use the Qwen3-VL-32B model

        Args:
            qwen_model: Preserved parameter (compatibility; not used)
            qwen_32b_model: Qwen-32B model name (Dashscope format)
            qwen_api_base: Dashscope API endpoint for Qwen
        """
        self.qwen_32b_model = qwen_32b_model  
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
    
    def call_openrouter_api(self, model, prompt, image, max_retries=3):  
        """Call OpenRouter API (used by UI-TARS)"""  
        base64_image = convert_pil_image_to_base64(image)  
        
        headers = {  
            "Authorization": f"Bearer {self.openrouter_api_key}",  
            "Content-Type": "application/json"  
        }  
        
        payload = {  
            "model": model,  
            "messages": [  
                {  
                    "role": "user",  
                    "content": [  
                        {"type": "text", "text": prompt},  
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}  
                    ]  
                }  
            ],  
            **self.override_generation_config  
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
                return result['choices'][0]['message']['content']  
            except Exception as e:  
                self.debug_print(f"OpenRouter API call failed (attempt {attempt + 1}/{max_retries}): {e}")  
                if attempt < max_retries - 1:  
                    time.sleep(2 ** attempt)  
        
        return None

    
    def plot_annotated_circle(self, image, point, radius=10, is_correct=None, alpha=100):    
        """Annotate predicted point on image - semi-transparent solid circle"""    
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))    
        draw = ImageDraw.Draw(overlay)    
          
        if point is not None:    
            x, y = point    
              
            if is_correct:    
                color = (0, 191, 255, alpha)  
            else:    
                color = (255, 0, 0, alpha)   
              
            draw.ellipse(    
                (x - radius, y - radius, x + radius, y + radius),    
                fill=color,    
                outline=None    
            )    
          
        annotated_image = image.convert('RGBA')    
        annotated_image = Image.alpha_composite(annotated_image, overlay)    
        return annotated_image.convert('RGB')   
        
    def load_model(self):    
        """Load model"""    
        pass    
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)  


    def call_dashscope_api(self, messages, model_name=None, max_retries=3):  
        """Call Dashscope API"""  
        headers = {  
            "Authorization": f"Bearer {self.dashscope_api_key}",  
            "Content-Type": "application/json"  
        }  
        
        payload = {    
                "model": model_name or self.qwen_model,  
                "messages": messages,    
                "temperature": 0.0 ,
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
        """Parse Dashscope normalized coordinates [0-1000] and convert to pixel coordinates"""    
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


    def ground_with_qwen_32b_initial(self, instruction, image):  
        """Layer 1: Qwen3-VL-32B initial check"""  
        self.debug_print("=== Layer 1: Qwen3-VL-32B initial check ===")  
        
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
        self.debug_print(f"Qwen-32B initial response: {response}")  
        
       
        pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)  
        
        return pixel_point, response, resized_image


    def refine_with_qwen(self, instruction, image, initial_pixel_point, initial_response, is_correct=False):    
        """Layer 2: Qwen3-VL-32B correction"""   
        self.debug_print("=== Layer 2: Qwen3-VL-32B correction ===")   
        resized_width, resized_height = image.size    
        
        if initial_pixel_point:    
               
            annotated_image = self.plot_annotated_circle(    
                image,    
                initial_pixel_point,    
                radius=10,    
                is_correct=is_correct,    
                alpha=100    
            )    
        else:    
            annotated_image = image    
        
        x_pixel, y_pixel = initial_pixel_point if initial_pixel_point else (resized_width//2, resized_height//2)    
        
          
        x_1000 = int((x_pixel / resized_width) * 1000)    
        y_1000 = int((y_pixel / resized_height) * 1000)    
        
       
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
        
      
        user_prompt = f"""Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

Core Requirements:
1. Red circle [{x_1000}, {y_1000}] (0-1000 scale) is the initial detection; only the bottom button area is valid.
2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

Output Format (STRICT):
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}
</tool_call>

x,y are integers [0,1000]. Output ONLY content within tags. Coordinates must lie in the bottom button area."""    
        
     
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
        self.debug_print(f"Qwen-32B correction response: {response}")   
        
        
        refined_pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)    
        
        return refined_pixel_point, response
    
    def refine_with_qwen_32b(self, instruction, image, first_pixel_point, second_pixel_point,   
                         first_response, second_response):  
        """Layer 3: Qwen3-VL-32B final correction"""  
        self.debug_print("=== Layer 3: Qwen3-VL-32B final correction ===")  
        resized_width, resized_height = image.size  
        
        
        annotated_image = image.copy()  
        
        # Mark Layer 1 result (red semi-transparent circle)  
        if first_pixel_point:  
            annotated_image = self.plot_annotated_circle(  
                annotated_image,  
                first_pixel_point,  
                radius=10,  
                is_correct=False,  
                alpha=100  
            )  
        
        # Mark Layer 2 result (blue semi-transparent circle)  
        if second_pixel_point:  
            annotated_image = self.plot_annotated_circle(  
                annotated_image,  
                second_pixel_point,  
                radius=10,  
                is_correct=True,    
                alpha=100  
            )  
        
        # Convert coordinates of both layers to 0–1000 normalized format  
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
        
     
        user_prompt = f"""FINAL VALIDATION: Locate "{instruction}" ONLY in the bottom button area

You see two marked points:
- RED CIRCLE: [{first_x_1000}, {first_y_1000}] (initial detection)
- BLUE CIRCLE: [{second_x_1000}, {second_y_1000}] (refined result)

Rules:
1. Only the bottom button area is valid
2. Choose the circle accurately centered on the target (must be in valid area)
3. If both are incorrect/off-area, find the target's true center in the bottom button area
4. Coordinates: integers [0,1000]

Output ONLY:
<tool_call>  
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}  
</tool_call> 
    
x,y integers [0,1000]."""  
        
     
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
        self.debug_print(f"Qwen-32B final correction response: {response}")  
        
      
        final_pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)  
        
        return final_pixel_point, response

    def ground_only_positive(self, instruction, image):  
        """Main entry: three-layer processing flow"""  
        self.logs = []  
        
     
        if isinstance(image, str):  
            image_path = image  
            image = Image.open(image_path).convert('RGB')  
        elif not isinstance(image, Image.Image):  
            raise ValueError("image must be a file path or PIL Image")  
            
        original_width, original_height = image.size  
        
         # Layer 1: Qwen3-VL-32B initial check  
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
            radius=10,    
            is_correct=False,  
            alpha=100    
        )    
        annotated_layer1.save("layer1_qwen32b_detection.png")    
        self.debug_print("Layer 1 completed")    
        
        time.sleep(1)    
        
        # Layer 2: Qwen-32B correction 
        qwen_32b_refined_pixel_point, qwen_32b_refined_response = self.refine_with_qwen(    
            instruction, resized_image, qwen_32b_pixel_point, qwen_32b_initial_response, is_correct=False    
        )    
        
        if qwen_32b_refined_pixel_point is None:    
            self.debug_print("Layer 2 correction failed; fallback to Layer 1 result")    
            qwen_32b_refined_pixel_point = qwen_32b_pixel_point    
        
        # Save Layer 2 annotated image  
        annotated_layer2 = self.plot_annotated_circle(    
            resized_image,    
            qwen_32b_refined_pixel_point,    
            radius=10,    
            is_correct=True,  
            alpha=100    
        )    
        annotated_layer2.save("layer2_qwen32b_refinement.png") 
        self.debug_print("Layer 2 completed")    
        
        time.sleep(1)    
        
        # Layer 3: Qwen-32B final correction  
        final_pixel_point, qwen_32b_final_response = self.refine_with_qwen_32b(    
            instruction, resized_image, qwen_32b_pixel_point, qwen_32b_refined_pixel_point,    
            qwen_32b_initial_response, qwen_32b_refined_response    
        )    
        
        # If Layer 3 fails, fallback to Layer 2  
        if final_pixel_point is None:    
            self.debug_print("Layer 3 correction failed; fallback to Layer 2 result")    
            final_pixel_point = qwen_32b_refined_pixel_point    
        
        # Save Layer 3 annotated image (show all three results)  
        annotated_final = resized_image.copy()    
        annotated_final = self.plot_annotated_circle(annotated_final, qwen_32b_pixel_point, 10, False, 100)    
        annotated_final = self.plot_annotated_circle(annotated_final, qwen_32b_refined_pixel_point, 10, True, 100)    
        annotated_final = self.plot_annotated_circle(annotated_final, final_pixel_point, 150, None, 120)    
        annotated_final.save("layer3_final_result.png")    
        self.debug_print("Layer 3 completed")    
        
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
                "qwen_32b_refined": qwen_32b_refined_response, 
                "qwen_32b_final": qwen_32b_final_response,    
                "logs": self.logs    
            }    
        }
    
    def ground_allow_negative(self, instruction, image):  
        """Grounding with support for negative samples — logic unchanged"""  
        return self.ground_only_positive(instruction, image)  
