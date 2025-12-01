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
    """UITars专用的smart_resize"""  
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

class UITarsQwenDualMethod: 
    def __init__(self,         
             uitars_model="bytedance/ui-tars-1.5-7b",    
             uitars_api_base="https://openrouter.ai/api/v1/chat/completions",    
             qwen_refine_model="qwen3-vl-32b-instruct",    
             qwen_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"):    
        """双层组合模型: UI-TARS初检(OpenRouter) + Qwen修正(Dashscope)"""          
        
        self.uitars_model = uitars_model    
        self.uitars_api_base = uitars_api_base    
        self.qwen_refine_model = qwen_refine_model    
        self.qwen_api_base = qwen_api_base   
        
       
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")    
        self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")    
        
        if not self.openrouter_api_key:    
            raise ValueError("请设置OPENROUTER_API_KEY环境变量")    
        if not self.dashscope_api_key:    
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")    
        
      
        
        self.uitars_client = OpenAI(    
            base_url=self.uitars_api_base.replace("/chat/completions", ""),    
            api_key=self.openrouter_api_key    
        )   
        
        self.override_generation_config = {        
            "temperature": 0.0,        
            "max_tokens": 2048,        
        }        
        
        self.logs = []        
        self.debug_flag = True
    
    def call_uitars_api(self, prompt, image, is_allow_negative=False):  
        """调用 OpenRouter 的 UITars API - 使用 OpenAI SDK"""  
        if isinstance(image, str):  
            image = Image.open(image).convert('RGB')  
        
        # 转换为 PNG 格式  
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
            
            self.debug_print(f"UITars API响应: {response_text}")  
            return response_text  
        except Exception as e:  
            self.debug_print(f"UITars API调用失败: {e}")  
            return None
    
    def extract_start_box_point(self, text):  
        """提取 start_box 或 point 格式的坐标"""  
        pattern_start_box = r"start_box='\((\d+),(\d+)\)'"  
        pattern_point = r"point='\((\d+),(\d+)\)'"  
        match = re.search(pattern_start_box, text, re.DOTALL)  
        if not match:  
            match = re.search(pattern_point, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2))]  
        return None  
    
    def extract_first_point(self, text):  
        """提取 [[x,y]] 格式的归一化坐标"""  
        pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
        match = re.search(pattern, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2))]  
        return None  
    
    def extract_first_bounding_box(self, text):  
        """提取 [[x1,y1,x2,y2]] 格式的边界框"""  
        pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"  
        match = re.search(pattern, text, re.DOTALL)  
        if match:  
            return [float(match.group(1)), float(match.group(2)),   
                    float(match.group(3)), float(match.group(4))]  
        return None


    def plot_annotated_circle(self, image, point, radius_outer=120, radius_inner=80, alpha=100):  
        """在图片上标注预测点 - 使用复合型半透明圆圈"""  
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  
        draw = ImageDraw.Draw(overlay)  
        
        if point is not None:  
            x, y = point  
            
            # 外圈: 红色半透明  
            draw.ellipse(  
                (x - radius_outer, y - radius_outer, x + radius_outer, y + radius_outer),  
                fill=(255, 0, 0, alpha),  # 红色  
                outline=None  
            )  
            
            # 内圈: 绿色半透明  
            draw.ellipse(  
                (x - radius_inner, y - radius_inner, x + radius_inner, y + radius_inner),  
                fill=(0, 255, 0, alpha),  # 绿色  
                outline=None  
            )  
        
        annotated_image = image.convert('RGBA')  
        annotated_image = Image.alpha_composite(annotated_image, overlay)  
        return annotated_image.convert('RGB')  
        
    def load_model(self):    
        """加载模型(API调用无需实际加载)"""    
        pass    
        
    def set_generation_config(self, **kwargs):    
        self.override_generation_config.update(kwargs)    
        
    def debug_print(self, string):    
        self.logs.append(string)    
        if self.debug_flag:    
            print(string)  


    def call_dashscope_api(self, messages, model_name=None, max_retries=3):  
        """调用Dashscope API (Qwen)"""  
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

    def parse_pixel_coordinates_raw(self, response_text, image):  
        """解析 UITars 返回的像素坐标"""  
        if not response_text:  
            return None  
        
        img_width, img_height = image.size  
        click_point = None  
    
        try:  
          
            start_box_point = self.extract_start_box_point(response_text)  
            if start_box_point:  
                self.debug_print(f"提取到start_box/point坐标: {start_box_point}")  
                
                new_height, new_width = smart_resize_uitars(img_height, img_width)  
                self.debug_print(f"图像分辨率适配: 原始({img_width}x{img_height}) -> 调整后({new_width}x{new_height})")  
               
                x = int(start_box_point[0] * img_width / new_width)  
                y = int(start_box_point[1] * img_height / new_height)  
                click_point = [x, y]  
        
            
            if not click_point:  
                first_point = self.extract_first_point(response_text)  
                if first_point:  
                    self.debug_print(f"提取到归一化点坐标: {first_point}")  
                    x = int(first_point[0] * img_width)  
                    y = int(first_point[1] * img_height)  
                    click_point = [x, y]  
        
         
            if not click_point:  
                bbox = self.extract_first_bounding_box(response_text)  
                if bbox:  
                    self.debug_print(f"提取到边界框: {bbox}")  
                    x = int((bbox[0] + bbox[2]) / 2 * img_width)  
                    y = int((bbox[1] + bbox[3]) / 2 * img_height)  
                    click_point = [x, y]  
    
     
            if click_point:  
                click_point[0] = max(0, min(img_width, click_point[0]))  
                click_point[1] = max(0, min(img_height, click_point[1]))  
                self.debug_print(f"最终有效像素坐标: {click_point}")  
                return click_point  
    
        except Exception as e:  
            self.debug_print(f"坐标解析失败: {e}")  
    
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


    def ground_with_uitars_initial(self, instruction, image):  
        """第一层: UI-TARS初检"""  
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
        
        return pixel_point, response, image  


    def refine_with_qwen(self, instruction, annotated_image, initial_pixel_point, initial_response, is_correct=False):  
        """第二层: Qwen3-VL修正"""  
        self.debug_print("=== 第二层: Qwen3-VL修正 ===")  
        
        resized_width, resized_height = annotated_image.size  
        x_pixel, y_pixel = initial_pixel_point if initial_pixel_point else (resized_width//2, resized_height//2)  
        
    
        x_1000 = int((x_pixel / resized_width) * 1000)  
        y_1000 = int((y_pixel / resized_height) * 1000)  
        
      
        prompt = f"""**CRITICAL: Ignore previous detection. Find the target independently.**  
    
    Task: {instruction}  
    
    The screenshot shows a composite marker at [{x_1000}, {y_1000}]:  
    - Outer red circle (larger radius): indicates the approximate detection area  
    - Inner green circle (smaller radius): marks the exact predicted center point  
    
    This is ONLY a reference marker from the initial detection.  
    
    Your job:  
    1. Independently find the correct element for: "{instruction}"  
    2. Output its center coordinates [x, y] where x,y ∈ [0,1000]  
    
    If the marked position (green center) is correct, output [{x_1000}, {y_1000}].  
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
            image = Image.open(image).convert('RGB')  
        elif not isinstance(image, Image.Image):  
            raise ValueError("image must be a file path or PIL Image")  
        
        original_width, original_height = image.size  
        
        # 第一层: UI-TARS初检  
        uitars_pixel_point, uitars_response, original_image = self.ground_with_uitars_initial(instruction, image)  
        
        if uitars_pixel_point is None:  
            return {  
                "result": "negative",  
                "point": None,  
                "bbox": None,  
                "raw_response": {"uitars_initial": uitars_response, "logs": self.logs}  
            }  
        
        # 创建带标注的图像用于第二层修正  
        annotated_image = self.plot_annotated_circle(  
            original_image,  
            uitars_pixel_point,  
            radius_outer=120,  
            radius_inner=80,    
            alpha=60  
        )  
        annotated_image.save("uitars_initial_detection.png")  
        self.debug_print("UI-TARS初检完成")  
        
        time.sleep(1)  # 避免API速率限制  
        
        # Qwen使用[0-1000]归一化坐标,需要转换  
        resized_height, resized_width = smart_resize(  
            original_height,  
            original_width,  
            factor=32,  
            min_pixels=3136,  
            max_pixels=2007040  
        )  
        resized_annotated_image = annotated_image.resize((resized_width, resized_height))  
        
        # 将UI-TARS的像素坐标映射到resized图像  
        scale_x = resized_width / original_width  
        scale_y = resized_height / original_height  
        resized_pixel_point = [  
            uitars_pixel_point[0] * scale_x,  
            uitars_pixel_point[1] * scale_y  
        ]  
        
        final_pixel_point, qwen_refined_response = self.refine_with_qwen(  
            instruction, resized_annotated_image, resized_pixel_point, uitars_response  
        )  
        
        # 修正失败回退  
        if final_pixel_point is None:  
            self.debug_print("Qwen修正失败, 回退到UI-TARS初检结果")  
            final_pixel_point = resized_pixel_point  
        
        # 坐标映射回原始图像  
        final_original_point = [  
            final_pixel_point[0] / scale_x,  
            final_pixel_point[1] / scale_y  
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
                "uitars_initial": uitars_response,  
                "qwen_refined": qwen_refined_response,  
                "logs": self.logs  
            }  
        }
    
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        return self.ground_only_positive(instruction, image)  

