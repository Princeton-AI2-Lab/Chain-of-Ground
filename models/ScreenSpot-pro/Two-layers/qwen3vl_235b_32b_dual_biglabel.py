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


class Qwen3VL235B32BDualMethod: 
    def __init__(self,     
             qwen_model="qwen3-vl-235b-a22b-instruct",  
             qwen_refine_model="qwen3-vl-32b-instruct", 
             qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"):  
        """      
        双层组合模型: Qwen初检(Dashscope) + Qwen修正(Dashscope)  
        
        Args:      
            qwen_model: Qwen初检模型名称(Dashscope格式)  
            qwen_refine_model: Qwen修正模型名称(Dashscope格式)  # 新增  
            qwen_api_base: Qwen的Dashscope API端点    
        """      
        self.qwen_model = qwen_model  
        self.qwen_refine_model = qwen_refine_model   
        self.qwen_api_base = qwen_api_base   
          
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")  
        if not self.dashscope_api_key:  
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")  
            
        self.override_generation_config = {    
            "temperature": 0.0,    
            "max_tokens": 2048,    
        }    
            
        self.logs = []    
        self.debug_flag = True  


    def plot_annotated_circle(self, image, point, radius=15, is_correct=None, alpha=100):    
        """在图片上标注预测点 - 使用半透明实心圆"""    
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))    
        draw = ImageDraw.Draw(overlay)    
          
        if point is not None:    
            x, y = point    
              
            if is_correct:    
                color = (0, 0, 255, alpha)    
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
        """加载模型"""    
        pass    
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)  


    def call_dashscope_api(self, messages, model_name=None, max_retries=3):  
        """调用Dashscope API"""  
        headers = {  
            "Authorization": f"Bearer {self.dashscope_api_key}",  
            "Content-Type": "application/json"  
        }  
        
        payload = {    
                "model": model_name or self.qwen_model,  
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
                    self.debug_print(f"检测到空<tool_call>标签 (尝试 {attempt + 1}/{max_retries})")  
                    if attempt < max_retries - 1:  
                        time.sleep(2 ** attempt)  
                        continue  
                
                self.debug_print(f"Qwen API响应状态: {response.status_code}")  
                self.debug_print(f"Qwen完整响应:\n{content}")  
                return content  
                
            except Exception as e:  
                self.debug_print(f"Qwen API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")  
                if attempt < max_retries - 1:  
                    time.sleep(2 ** attempt)  
        
        return None  


    def parse_normalized_coordinates(self, response, resized_width, resized_height):    
        """解析Dashscope返回的归一化坐标[0-1000]并转换为像素坐标"""    
        if not response:    
            self.debug_print("响应为空")  
            return None    
            
        try:    
        
            if '<tool_call>' not in response:  
                self.debug_print(f"未找到<tool_call>标签,原始响应: {response[:200]}")  
               
                try:  
                 
                    coord_match = re.search(r'"coordinate"\s*:\s*\[(\d+),\s*(\d+)\]', response)  
                    if coord_match:  
                        x_1000 = int(coord_match.group(1))  
                        y_1000 = int(coord_match.group(2))  
                        self.debug_print(f"通过回退逻辑提取坐标: [{x_1000}, {y_1000}]")  
                        
                     
                        x_pixel = (x_1000 / 1000.0) * resized_width    
                        y_pixel = (y_1000 / 1000.0) * resized_height    
                        self.debug_print(f"归一化坐标: [{x_1000}, {y_1000}] -> 像素坐标: [{x_pixel:.1f}, {y_pixel:.1f}]")  
                        return [x_pixel, y_pixel]  
                    else:  
                        self.debug_print("回退逻辑也无法提取坐标,返回 None")  
                        return None  
                except Exception as e:  
                    self.debug_print(f"回退逻辑失败: {e}")  
                    return None  
            
          
            tool_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)  
            if not tool_match:  
                self.debug_print("找到<tool_call>标签但无法提取内容")  
                return None  
                
            tool_json = tool_match.group(1).strip()  
            data = json.loads(tool_json)  
            coords = data.get("arguments", {}).get("coordinate", [])  
                
            if len(coords) == 2:    
               
                x_normalized = float(coords[0])    
                y_normalized = float(coords[1])    
                    
            
                x_pixel = (x_normalized / 1000.0) * resized_width    
                y_pixel = (y_normalized / 1000.0) * resized_height    
                    
                self.debug_print(f"归一化坐标: [{x_normalized}, {y_normalized}] -> 像素坐标: [{x_pixel:.1f}, {y_pixel:.1f}]")    
                return [x_pixel, y_pixel]    
                
            self.debug_print("坐标格式不正确")    
            return None    
                
        except json.JSONDecodeError as e:    
            self.debug_print(f"JSON解析失败: {e}")  
     
            try:  
                coord_match = re.search(r'\[(\d+),\s*(\d+)\]', response)  
                if coord_match:  
                    x_1000 = int(coord_match.group(1))  
                    y_1000 = int(coord_match.group(2))  
                    self.debug_print(f"JSON解析失败,通过正则提取: [{x_1000}, {y_1000}]")  
                    x_pixel = (x_1000 / 1000.0) * resized_width    
                    y_pixel = (y_1000 / 1000.0) * resized_height  
                    return [x_pixel, y_pixel]  
            except Exception as fallback_e:  
                self.debug_print(f"回退解析也失败: {fallback_e}")  
        except Exception as e:    
            self.debug_print(f"坐标解析失败: {e}")    
            
        return None 


    def normalize_pixel_coordinates(self, pixel_point, width, height):    
        """将像素坐标归一化到[0,1]范围"""    
        if pixel_point is None:    
            return None    
            
        try:    
            x_pixel, y_pixel = pixel_point    
            x_norm = max(0.0, min(1.0, x_pixel / width))    
            y_norm = max(0.0, min(1.0, y_pixel / height))    
            self.debug_print(f"归一化: [{x_pixel}, {y_pixel}] -> [{x_norm:.4f}, {y_norm:.4f}]")    
            return [x_norm, y_norm]    
        except Exception as e:    
            self.debug_print(f"归一化失败: {e}")    
            return None


    def ground_with_qwen_initial(self, instruction, image):    
        """第一层: Qwen3-VL初检"""    
        input_width, input_height = image.size    
        
        
        resized_height, resized_width = smart_resize(    
            input_height,    
            input_width,    
            factor=32,    
            min_pixels=3136,    
            max_pixels=2007040    
        )    
        resized_image = image.resize((resized_width, resized_height))    
        
        self.debug_print(f"图像缩放: {input_width}x{input_height} -> {resized_width}x{resized_height}")    
        
     
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
    
        response = self.call_dashscope_api(messages, model_name=self.qwen_model)    
        
     
        pixel_point = self.parse_normalized_coordinates(response, resized_width, resized_height)    
        
        return pixel_point, response, resized_image  


    def refine_with_qwen(self, instruction, annotated_image, initial_pixel_point, initial_response, is_correct=False):          
        """第二层: Qwen3-VL修正"""          
        self.debug_print("=== 第二层: Qwen3-VL修正 ===")          
        resized_width, resized_height = annotated_image.size  
        
        x_pixel, y_pixel = initial_pixel_point if initial_pixel_point else (resized_width//2, resized_height//2)          
        
           
        x_1000 = int((x_pixel / resized_width) * 1000)        
        y_1000 = int((y_pixel / resized_height) * 1000)        
        
     
        prompt = f"""**CRITICAL: Ignore previous detection. Find the target independently.**  
    
    Task: {instruction}  
    
    The screenshot shows a red circle at [{x_1000}, {y_1000}]. This is ONLY a reference marker.  
    
    Your job:  
    1. Find the correct element for: "{instruction}"  
    2. Output its center coordinates [x, y] where x,y ∈ [0,1000]  
    
    If the red circle happens to be correct, output [{x_1000}, {y_1000}].  
    If not, output the CORRECT coordinates.  
    
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
        self.debug_print(f"Qwen修正响应: {refined_response}")          
        
        refined_pixel_point = self.parse_normalized_coordinates(refined_response, resized_width, resized_height)          
        
        return refined_pixel_point, refined_response

    def ground_only_positive(self, instruction, image):    
        """主入口"""    
        self.logs = []    
        
        # 加载原始图像  
        if isinstance(image, str):    
            image_path = image    
            image = Image.open(image_path).convert('RGB')    
        elif not isinstance(image, Image.Image):    
            raise ValueError("image must be a file path or PIL Image")    
            
        original_width, original_height = image.size    
        
        # 第一层: Qwen3-VL初检  
        qwen_pixel_point, qwen_response, resized_image = self.ground_with_qwen_initial(instruction, image)    
        resized_width, resized_height = resized_image.size    
        
        # 创建带标注的图像用于第二层修正  
        if qwen_pixel_point:    
            annotated_image = self.plot_annotated_circle(    
                resized_image,    
                qwen_pixel_point,    
                radius=100,    
                is_correct=False,  
                alpha=60    
            )    
            annotated_image.save("qwen_initial_detection.png")    
            self.debug_print("初检完成")    
        else:  
            return {    
                "result": "negative",    
                "point": None,    
                "bbox": None,    
                "raw_response": {"qwen_initial": qwen_response, "logs": self.logs}    
            }    
        
        time.sleep(1)  # 避免API速率限制    
    
        # 第二层: 调用Qwen修正 - 传入带标注的图像  
        final_pixel_point, qwen_refined_response = self.refine_with_qwen(    
            instruction, annotated_image, qwen_pixel_point, qwen_response, is_correct=False    
        )    
    
        # 修正失败回退  
        if final_pixel_point is None:    
            self.debug_print("Qwen修正失败, 回退到Qwen初检结果")    
            final_pixel_point = qwen_pixel_point    
        
        # 坐标映射回原始图像  
        scale_x = original_width / resized_width    
        scale_y = original_height / resized_height    
        final_original_point = [    
            final_pixel_point[0] * scale_x,    
            final_pixel_point[1] * scale_y    
        ]    
        
        # 归一化到[0,1]  
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
                "qwen_refined": qwen_refined_response,    
                "logs": self.logs    
            }    
        }
    
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)  

