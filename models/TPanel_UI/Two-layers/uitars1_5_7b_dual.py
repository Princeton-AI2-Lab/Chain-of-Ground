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
  
  
class UITarsDualMethod:  
    def __init__(self, uitars_model="bytedance/ui-tars-1.5-7b",  
                 api_base="https://openrouter.ai/api/v1",
                 http_referer=None,
                 x_title=None):  
        """
        双层UI-TARS自我修正模型: UI-TARS初检 + UI-TARS修正  
        ual
        Args:  
            uitars_model: UI-TARS模型名称  
            api_base: OpenRouter API端点  
            http_referer: 可选
            x_title: 可选
        """  
        self.uitars_model = uitars_model  
        self.api_base = api_base   
          
       
        self.api_key = os.environ.get("OPENROUTER_API_KEY")  
        if not self.api_key:  
            raise ValueError("请设置OPENROUTER_API_KEY环境变量")  
          
    
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
        """加载模型"""  
        pass  
      
    def set_generation_config(self, **kwargs):  
        """设置生成配置"""
        
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        
        supported_params = ["temperature", "top_p", "max_tokens", "stop", "n"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
        
        self.override_generation_config.update(filtered_kwargs)  
        self.debug_print(f"更新生成配置: {self.override_generation_config}")
      
    def debug_print(self, string):  
        self.logs.append(string)  
        if self.debug_flag:  
            print(string) 

    def plot_red_transparent_circle(self, image, point, radius=80, alpha=60):  
        """在图像上绘制单红色半透明圆标注"""  
        if point is None:  
            return image  
            
        from PIL import ImageDraw  
          
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))  
        draw = ImageDraw.Draw(overlay)  
        
        x, y = point  
       
        draw.ellipse(  
            (x - radius, y - radius, x + radius, y + radius),  
            fill=(255, 0, 0, alpha),  
            outline=(255, 0, 0, 200)  
        )  
        
        annotated_image = image.convert('RGBA')  
        annotated_image = Image.alpha_composite(annotated_image, overlay)  
        return annotated_image.convert('RGB')  

    def call_uitars_api(self, prompt, image, is_allow_negative=False):
        """调用OpenRouter的UI-TARS API"""
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
            self.debug_print(f"\nOpenRouter API响应完成: {response_text}")
            return response_text

        except TypeError as e:
            
            if "unexpected keyword argument 'max_new_tokens'" in str(e):
                error_msg = "错误：OpenRouter API不支持max_new_tokens参数，请使用max_tokens替代！"
                self.debug_print(error_msg)
                raise ValueError(error_msg) from e
            else:
                self.debug_print(f"OpenRouter调用失败（参数错误）: {e}")
                return None
        except BadRequestError as e:
            self.debug_print(f"OpenRouter调用失败（请求无效）: {e}")
            return None
        except Exception as e:
            self.debug_print(f"OpenRouter调用失败（网络/权限问题）: {e}")
            return None

    def parse_pixel_coordinates_raw(self, response_text, image):
        """解析像素坐标"""
        if not response_text:
            return None
        
        img_width, img_height = image.size
        click_point = None

        try:
           
            start_box_point = extract_start_box_point(response_text)
            if start_box_point:
                self.debug_print(f"提取到start_box/point坐标: {start_box_point}")
                
                new_height, new_width = smart_resize(img_height, img_width)
                self.debug_print(f"图像分辨率适配: 原始({img_width}x{img_height}) -> 调整后({new_width}x{new_height})")
             
                x = int(start_box_point[0] / new_width * img_width)
                y = int(start_box_point[1] / new_height * img_height)
                click_point = [x, y]
        
          
            if not click_point:
                first_point = extract_first_point(response_text)
                if first_point:
                    self.debug_print(f"提取到归一化点坐标: {first_point}")
                  
                    x = int(first_point[0] * img_width)
                    y = int(first_point[1] * img_height)
                    click_point = [x, y]
    
            if not click_point:
                bbox = extract_first_bounding_box(response_text)
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

    def normalize_pixel_coordinates(self, pixel_point, width, height):  
        """将像素坐标归一化到 [0,1] 范围"""  
        if pixel_point is None:  
            return None  
          
        try:  
            x_pixel, y_pixel = pixel_point  
            x_norm = max(0.0, min(1.0, x_pixel / width))  
            y_norm = max(0.0, min(1.0, y_pixel / height))  
            self.debug_print(f"坐标归一化: [{x_pixel}, {y_pixel}] -> [{x_norm:.4f}, {y_norm:.4f}]")  
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
    
        return pixel_point, response
    
    def refine_with_uitars(self, instruction, image, initial_pixel_point, initial_response):      
        """第二层: UITars自我修正"""      
        self.debug_print("=== 第二层: UITars自我修正 ===")      
        
        width, height = image.size      
        
    
        annotated_image = self.plot_red_transparent_circle(      
            image,       
            initial_pixel_point,       
            radius=10,    
            alpha=100    
        )      
        
          
        try:      
            annotated_image.save("layer1_initial_detection.png")      
            self.debug_print("第一层标注图像已保存: layer1_initial_detection.png")      
        except Exception as e:      
            self.debug_print(f"保存标注图像失败: {e}")      
        
      
        prompt = f"""Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

    The screenshot shows a RED TRANSPARENT CIRCLE — this is the initial detection reference; only the bottom button area is valid.

    Your job:
    1. Independently verify the target element for "{instruction}"
    2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
    3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

    Output Format (STRICT):
    Action: click(point='<point>x1 y1</point>')

    NO explanations. Output ONLY the action line with pixel coordinates."""      
        
            
        response = self.call_uitars_api(prompt, annotated_image)      
        
        
        refined_pixel_point = self.parse_pixel_coordinates_raw(response, annotated_image)      
        
        return refined_pixel_point, response

    
    def ground_only_positive(self, instruction, image):      
        """主入口: 双层UI-TARS检测(初检+修正)"""      
        self.logs = []      
        
        if isinstance(image, str):      
            image = Image.open(image).convert('RGB')      
        
        width, height = image.size      
        
        # 第一层: UI-TARS初检  
        self.debug_print("=== 第一层: UI-TARS初检 ===")  
        uitars_pixel_point, uitars_initial_response = self.ground_with_uitars_initial(instruction, image)      
        
        if uitars_pixel_point is None:      
            return {      
                "result": "negative",      
                "point": None,      
                "bbox": None,      
                "raw_response": {"uitars_initial": uitars_initial_response, "logs": self.logs}      
            }      
        
        self.debug_print(f"第一层初检坐标: {uitars_pixel_point}")  
        
        # 第二层: UITars自我修正  
        refined_pixel_point, uitars_refined_response = self.refine_with_uitars(  
            instruction,   
            image,   
            uitars_pixel_point,   
            uitars_initial_response  
        )  
        
        # 如果第二层修正失败,回退到第一层结果  
        if refined_pixel_point is None:  
            self.debug_print("第二层修正失败,回退到第一层结果")  
            final_pixel_point = uitars_pixel_point  
        else:  
            self.debug_print(f"第二层修正坐标: {refined_pixel_point}")  
            final_pixel_point = refined_pixel_point  
        
        # 归一化最终结果  
        final_normalized_point = self.normalize_pixel_coordinates(final_pixel_point, width, height)      
        
        return {      
            "result": "positive" if final_normalized_point else "negative",      
            "point": final_normalized_point,      
            "bbox": None,      
            "raw_response": {      
                "uitars_initial": uitars_initial_response,  
                "uitars_refined": uitars_refined_response,  
                "logs": self.logs      
            }      
        }
      
    def ground_allow_negative(self, instruction, image):  
        """支持负样本的grounding"""  
        self.logs = []    
        
        if isinstance(image, str):    
            image = Image.open(image).convert('RGB')    
        
        width, height = image.size  

        # 负样本专用提示词
        prompt = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1.
If such element does not exist, output only the text 'Target not existent'.
The instruction is:
{instruction}"""

        # 调用OpenRouter API（允许负样本判断）
        response = self.call_uitars_api(prompt, image, is_allow_negative=True)
        
        # 检查负样本标识（不区分大小写）
        if response and "target not existent" in response.lower():
            self.debug_print("检测到负样本: 目标元素不存在")
            return {
                "result": "negative",
                "point": None,
                "bbox": None,
                "raw_response": {"uitars_initial": response, "logs": self.logs}
            }

        # 解析坐标
        pixel_point = self.parse_pixel_coordinates_raw(response, image)  
        if pixel_point is None:    
            return {    
                "result": "negative",    
                "point": None,    
                "bbox": None,    
                "raw_response": {"uitars_initial": response, "logs": self.logs}    
            }  

        # 归一化坐标
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
