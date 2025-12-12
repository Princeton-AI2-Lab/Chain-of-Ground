import os
import sys
sys.path.append(os.path.dirname(__file__))
import argparse
import json
import copy
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image, ImageDraw
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from backends.qwen_backend import QwenBackend
from backends.uitars_backend import UITarsBackend, smart_resize as smart_resize_uitars
import core.prompts as prompts

def convert_pil_image_to_base64(image, format="JPEG"):
    b = BytesIO()
    image.save(b, format=format)
    return base64.b64encode(b.getvalue()).decode()

def plot_circle(image, point, radius=10, color=(255,0,0,100)):
    overlay = Image.new('RGBA', image.size, (255,255,255,0))
    d = ImageDraw.Draw(overlay)
    if point is not None:
        x,y = point
        d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=None)
    im = image.convert('RGBA')
    im = Image.alpha_composite(im, overlay)
    return im.convert('RGB')

def plot_box(image, point, box_size=100, color=(0,0,255,80)):
    overlay = Image.new('RGBA', image.size, (255,255,255,0))
    d = ImageDraw.Draw(overlay)
    if point is not None:
        x,y = point
        h = box_size//2
        d.rectangle((x-h, y-h, x+h, y+h), fill=color, outline=None)
    im = image.convert('RGBA')
    im = Image.alpha_composite(im, overlay)
    return im.convert('RGB')

def is_uitars_model(name):
    if not name:
        return False
    return 'ui-tars' in name or 'bytedance/ui-tars' in name

def build_qwen_system():
    return [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": """
    # Tools
    You may call one or more functions to assist with the user query.
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type": "function", "function": {"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* The screen's resolution is 1000x1000.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button.", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.", "type": "array"}}, "required": ["action", "coordinate"], "type": "object"}}}
    </tools>
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>"""}
    ]

def stage1(instruction, image, model_name):
    if is_uitars_model(model_name):
        iw, ih = image.size
        rh, rw = smart_resize_uitars(ih, iw)
        resized = image.resize((rw, rh))
        backend = UITarsBackend(model_name=model_name)
        prompt = prompts.INITIAL_UITARS.format(instruction=instruction)
        resp = backend.call(prompt, resized, model_name=model_name)
        pt = backend.parse_pixel_coordinates(resp, resized)
        return pt, resp, resized
    else:
        iw, ih = image.size
        rh, rw = smart_resize(ih, iw, factor=32, min_pixels=3136, max_pixels=2007040)
        resized = image.resize((rw, rh))
        backend = QwenBackend(model_name=model_name)
        user_prompt = prompts.INITIAL_QWEN.format(instruction=instruction)
        b64 = convert_pil_image_to_base64(resized)
        messages = [
            {"role": "system", "content": build_qwen_system()},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": user_prompt}]}
        ]
        resp = backend.call(messages, model_name=model_name)
        pt = backend.parse_normalized_coordinates(resp, rw, rh)
        return pt, resp, resized

def stage2(instruction, annotated_image, initial_point, model_name):
    if is_uitars_model(model_name):
        # UITars branch
        iw, ih = annotated_image.size
        rh, rw = smart_resize_uitars(ih, iw)
        resized = annotated_image.resize((rw, rh))
        
        backend = UITarsBackend(model_name=model_name)
        prompt = prompts.REFINE_UITARS_CIRCLE_TPanel_UI.format(instruction=instruction)
        resp = backend.call(prompt, resized, model_name=model_name)
        pt = backend.parse_pixel_coordinates(resp, resized)
        return pt, resp, resized
    else:
        # Qwen branch
        iw, ih = annotated_image.size
        rh, rw = smart_resize(ih, iw, factor=32, min_pixels=3136, max_pixels=2007040)
        resized = annotated_image.resize((rw, rh))
        
        # Convert pixel coordinates to [0-1000] range
        x_1000 = int((initial_point[0] / rw) * 1000) if initial_point else 500
        y_1000 = int((initial_point[1] / rh) * 1000) if initial_point else 500
        
        backend = QwenBackend(model_name=model_name)
        user_prompt = prompts.REFINE_QWEN_CIRCLE.format(
            instruction=instruction,
            x_1000=x_1000,
            y_1000=y_1000
        )
        b64 = convert_pil_image_to_base64(resized)
        messages = [
            {"role": "system", "content": build_qwen_system()},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        resp = backend.call(messages, model_name=model_name)
        pt = backend.parse_normalized_coordinates(resp, rw, rh)
        return pt, resp, resized

def stage3_final_compare(instruction, image, first_point, second_point, model_name):
    iw, ih = image.size
    fx = int((first_point[0] / iw) * 1000) if first_point else 500
    fy = int((first_point[1] / ih) * 1000) if first_point else 500
    sx = int((second_point[0] / iw) * 1000) if second_point else 500
    sy = int((second_point[1] / ih) * 1000) if second_point else 500
    annotated = image.copy()
    annotated = plot_circle(annotated, first_point, radius=100, color=(255,0,0,60))
    annotated = plot_box(annotated, second_point, box_size=100, color=(0,0,255,80))
    backend = QwenBackend(model_name=model_name)
    user_prompt = prompts.FINAL_QWEN_BLUEBOX.format(instruction=instruction, first_x_1000=fx, first_y_1000=fy, second_x_1000=sx, second_y_1000=sy)
    b64 = convert_pil_image_to_base64(annotated)
    messages = [
        {"role": "system", "content": build_qwen_system()},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}, {"type": "text", "text": user_prompt}]}
    ]
    resp = backend.call(messages, model_name=model_name)
    pt = backend.parse_normalized_coordinates(resp, iw, ih)
    return pt, resp

def normalize_point(point, w, h):
    if point is None:
        return None
    return [max(0.0, min(1.0, point[0] / w)), max(0.0, min(1.0, point[1] / h))]

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']

def eval_sample_positive_gt(sample, point_norm):
    bbox = sample["bbox"]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    if point_norm is None:
        return "wrong_format"
    if (bbox[0] <= point_norm[0] <= bbox[2]) and (bbox[1] <= point_norm[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"

def eval_sample_negative_gt(point_norm):
    if point_norm is None:
        return "correct"
    return "wrong"

def calc_metric_for_result_list(results):
    n = len(results)
    c = sum(1 for r in results if r["correctness"] == "correct")
    wf = sum(1 for r in results if r["correctness"] == "wrong_format")
    tr = [r for r in results if r.get("ui_type") == "text"]
    ir = [r for r in results if r.get("ui_type") == "icon"]
    tc = sum(1 for r in tr if r["correctness"] == "correct")
    ic = sum(1 for r in ir if r["correctness"] == "correct")
    return {"num_correct_action": c, "num_total": n, "wrong_format_num": wf, "action_acc": c / n if n > 0 else 0, "text_acc": tc / len(tr) if len(tr) > 0 else 0, "icon_acc": ic / len(ir) if len(ir) > 0 else 0}

def run_batch(args):

    
    # Task collection
    if args.task == "all":
        task_files = [os.path.splitext(f)[0] for f in os.listdir(args.tests_dir) if f.endswith('.json')]
    else:
        task_files = args.task.split(',')
    
    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(',')
    
    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(',')
    
    tasks_to_run = []
    for tf in task_files:
        with open(os.path.join(args.tests_dir, tf + ".json"), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for inst in inst_styles:
            for gt in gt_types:
                for s in data:
                    x = copy.deepcopy(s)
                    x["task_filename"] = tf
                    x["gt_type"] = gt
                    x["instruction_style"] = inst
                    x["prompt_to_evaluate"] = x.get("instruction")
                    tasks_to_run.append(x)
    
    print(f"Total tasks to run: {len(tasks_to_run)}")
    
    # Visualization configuration
    VISUALIZE_FIRST_N = 100
    vis_count = 0
    model1_name = args.model1.replace('/', '_').replace(':', '_')
    model2_name = args.model2.replace('/', '_').replace(':', '_')
    vis_dir = f"./visualizations_{args.mode}_{model1_name}_{model2_name}"
    os.makedirs(vis_dir, exist_ok=True)
    
    results = []
    for sample in tqdm(tasks_to_run, desc="Processing samples"):
        img_path = os.path.join(args.screens_dir, sample["img_filename"])
        img = Image.open(img_path).convert('RGB')
        
        # Fill missing fields
        if "img_size" not in sample:
            sample["img_size"] = [img.width, img.height]
        
        if "bbox" in sample and len(sample["bbox"]) == 4:
            x, y, w, h = sample["bbox"][0], sample["bbox"][1], sample["bbox"][2], sample["bbox"][3]
            if x < w and y < h:
                sample["bbox"] = [x, y, x + w, y + h]
        
        if "data_type" in sample and "ui_type" not in sample:
            sample["ui_type"] = sample["data_type"]
        
        if "data_source" in sample and "platform" not in sample:
            sample["platform"] = sample["data_source"]
        
        if "id" not in sample:
            sample["id"] = sample["img_filename"].rsplit('.', 1)[0]
        
        if "group" not in sample:
            sample["group"] = "unknown"
        
        if "application" not in sample:
            sample["application"] = sample.get("data_source", "unknown")
        
        # Stage 1
        p1, r1, resized = stage1(sample["prompt_to_evaluate"], img, args.model1)
        
        if sample["gt_type"] == "positive" and p1 is None:
            results.append({
                "id": sample["id"],
                "img_path": img_path,
                "group": sample.get("group"),
                "platform": sample.get("platform"),
                "application": sample.get("application"),
                "lang": sample.get("language"),
                "instruction_style": sample.get("instruction_style"),
                "prompt_to_evaluate": sample["prompt_to_evaluate"],
                "gt_type": sample["gt_type"],
                "ui_type": sample.get("ui_type"),
                "task_filename": sample.get("task_filename"),
                "pred": None,
                "raw_response": {"stage1": r1},
                "bbox": sample.get("bbox"),
                "correctness": "wrong_format"
            })
            continue
        
        # Stage 2
        ann1 = plot_circle(resized, p1, radius=100, color=(255,0,0,60))
        p2, r2, resized2 = stage2(sample["prompt_to_evaluate"], ann1, p1, args.model2)
        
        final_pt = p2 if p2 is not None else p1
        rr = {"stage1": r1, "stage2": r2}
        
        # Stage 3 (if triple mode)
        if args.mode == 'triple':
            p3, r3 = stage3_final_compare(sample["prompt_to_evaluate"], resized2, p1, final_pt, args.model3)
            final_pt = p3 if p3 is not None else final_pt
            rr["stage3"] = r3
        
        # Coordinate conversion: resized -> original
        rw, rh = resized2.size
        sx = img.width / rw
        sy = img.height / rh
        
        if final_pt is not None:
            fo = [final_pt[0] * sx, final_pt[1] * sy]
            pn = normalize_point(fo, img.width, img.height)
        else:
            fo = None
            pn = None
        
        # Evaluate correctness
        if sample["gt_type"] == "positive":
            corr = eval_sample_positive_gt(sample, pn)
        else:
            corr = eval_sample_negative_gt(pn)
        
        # Visualization (first 100 positive samples)
        if vis_count < VISUALIZE_FIRST_N and sample["gt_type"] == "positive" and fo is not None:
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Draw ground truth bbox (green)
            bbox = sample["bbox"]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=3,
                edgecolor='green',
                facecolor='none',
                label='Ground Truth'
            )
            ax.add_patch(rect)
            
            # Draw prediction point (blue=correct, red=wrong)
            color = 'blue' if corr == 'correct' else 'red'
            ax.plot(fo[0], fo[1], 'o', color=color, markersize=12,
                   label=f'Prediction ({corr})')
            
            # Title
            title = f"Sample {vis_count + 1}/{VISUALIZE_FIRST_N}\n"
            title += f"Task: {sample['task_filename']} | Platform: {sample['platform']}\n"
            title += f"Instruction: {sample['prompt_to_evaluate'][:80]}...\n"
            title += f"Correctness: {corr}"
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.axis('off')
            
            # Save
            plt.savefig(f"{vis_dir}/sample_{vis_count:04d}_{corr}.png",
                       bbox_inches='tight', dpi=100)
            plt.close()
            
            vis_count += 1
        
        # Record results
        results.append({
            "id": sample["id"],
            "img_path": img_path,
            "group": sample.get("group"),
            "platform": sample.get("platform"),
            "application": sample.get("application"),
            "lang": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample["gt_type"],
            "ui_type": sample.get("ui_type"),
            "task_filename": sample.get("task_filename"),
            "pred": fo if final_pt is not None else None,
            "raw_response": rr,
            "bbox": sample.get("bbox") if sample["gt_type"] == "positive" else None,
            "correctness": corr
        })
    
    # Calculate overall metrics
    metrics = calc_metric_for_result_list(results)
    out = {
        "metrics": {"overall": metrics},
        "details": results
    }
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"\n✅ Results saved to: {args.output}")
        print(f"✅ Visualizations saved to: {vis_dir}/ ({vis_count} images)")
    else:
        print(json.dumps(out, indent=2))
    
    # Print overall metrics
    print(f"\n📊 Overall Metrics:")
    print(f"  - Accuracy: {metrics['action_acc']:.2%}")
    print(f"  - Correct: {metrics['num_correct_action']}/{metrics['num_total']}")
    print(f"  - Text Acc: {metrics['text_acc']:.2%}")
    print(f"  - Icon Acc: {metrics['icon_acc']:.2%}")
    print(f"  - Wrong Format: {metrics['wrong_format_num']}")

def run(args):
    img = Image.open(args.image).convert('RGB')
    iw, ih = img.size
    
    # Stage 1
    p1, r1, resized = stage1(args.instruction, img, args.model1)
    if p1 is None:
        print(json.dumps({"result": "negative", "point": None, "raw_response": {"stage1": r1}}))
        return
    
    # Stage 2: Draw large red circle
    ann1 = plot_circle(resized, p1, radius=100, color=(255,0,0,60))
    p2, r2, resized2 = stage2(args.instruction, ann1, p1, args.model2)
    
    if args.mode == 'dual':
        # Dual mode: Use stage2 result
        rw, rh = resized2.size
        sx = iw / rw
        sy = ih / rh
        fp = p2 if p2 is not None else p1
        fo = [fp[0] * sx, fp[1] * sy]
        pn = normalize_point(fo, iw, ih)
        print(json.dumps({
            "result": "positive" if pn else "negative",
            "point": pn,
            "raw_response": {"stage1": r1, "stage2": r2}
        }))
        return
    
    # Triple mode
    p3, r3 = stage3_final_compare(args.instruction, resized2, p1, p2 if p2 is not None else p1, args.model3)
    rw, rh = resized2.size
    sx = iw / rw
    sy = ih / rh
    fp = p3 if p3 is not None else (p2 if p2 is not None else p1)
    fo = [fp[0] * sx, fp[1] * sy]
    pn = normalize_point(fo, iw, ih)
    print(json.dumps({
        "result": "positive" if pn else "negative",
        "point": pn,
        "raw_response": {"stage1": r1, "stage2": r2, "stage3": r3}
    }))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['dual','triple'], required=True)
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--model3', type=str)
    parser.add_argument('--instruction', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--dataset_type', type=str, choices=['sspro','tpanelui'])
    parser.add_argument('--screens_dir', type=str)
    parser.add_argument('--tests_dir', type=str)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--inst_style', type=str, default='instruction')
    
    parser.add_argument('--gt_type', type=str, default='positive')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    if args.mode == 'triple' and not args.model3:
        raise ValueError('model3 required for triple mode')
    if args.batch:
        if not args.screens_dir or not args.tests_dir:
            raise ValueError('screens_dir and tests_dir required for batch mode')
        run_batch(args)
        return
    run(args)

if __name__ == '__main__':
    main()
