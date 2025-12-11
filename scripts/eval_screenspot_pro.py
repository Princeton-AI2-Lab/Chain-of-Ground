import copy
import itertools

import torch
import json
import re
import argparse
import os
import sys
sys.path.append(os.path.dirname(__file__))
from PIL import Image
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

from model_factory import build_model

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['dual','triple'], required=True)
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--model3', type=str)
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--dataset_type', type=str, choices=['sspro','tpanelui'])
    parser.add_argument('--screens_dir', type=str)
    parser.add_argument('--tests_dir', type=str)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--inst_style', type=str, default='instruction')
    parser.add_argument('--gt_type', type=str, default='positive')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args


def collect_results_to_eval(results, platform=None, group=None, application=None, gt_type=None, instruction_style=None, ui_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, gt_type=False, instruction_style=False, ui_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]  # may be none
    print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        application=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        application = combo.get("application")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            application=application,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} app:{application} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        application=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        application = combo.get("application")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            application=application,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"app:{application}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        group=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        group = combo.get("group")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            group=group,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"group:{group}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report

def run_cli_aligned(args):
    import importlib.util
    root = os.path.dirname(os.path.dirname(__file__))
    cli_main_path = os.path.join(root, 'cli', 'main.py')
    spec = importlib.util.spec_from_file_location('cli_main', cli_main_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if args.mode == 'triple' and not args.model3:
        raise ValueError('model3 required for triple mode')
    if args.batch:
        if not args.screens_dir or not args.tests_dir:
            raise ValueError('screens_dir and tests_dir required for batch mode')
        mod.run_batch(args)
        return
    mod.run(args)

def main(args):
    model = build_model(args)
    print("Load model success")

    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r',encoding='utf-8') as f:
            task_data = json.load(f)

        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for task_instance in task_data:
                    task_instance = copy.deepcopy(task_instance)
                    task_instance["task_filename"] = task_filename
                    task_instance["gt_type"] = gt_type
                    task_instance["instruction_style"] = inst_style
                    if "language" not in task_instance:
                        task_instance["language"] = "en"
                    task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                    tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} = {len(task_data) * len(inst_styles) * len(gt_types)}")
    print(f"Total tasks: {len(tasks_to_run)}")

 
    VISUALIZE_FIRST_N = 100
    vis_count = 0
    vis_dir = f"./visualizations_{args.model_type}"
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    for sample in tqdm(tasks_to_run):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)

          
        if "img_size" not in sample:    
            img = Image.open(img_path)  
            sample["img_size"] = [img.width, img.height] 

        
        
        if "bbox" in sample and len(sample["bbox"]) == 4:  
            x, y, w, h = sample["bbox"]  
            sample["bbox"] = [x, y, x + w, y + h]  
        
      
        if "data_type" in sample and "ui_type" not in sample:  
            sample["ui_type"] = sample["data_type"]  
        if "data_source" in sample and "platform" not in sample:  
            sample["platform"] = sample["data_source"] 

         
        if "id" not in sample:  
            
            sample["id"] = filename.rsplit('.', 1)[0]    
        
         
        if "group" not in sample:  
            sample["group"] = "unknown"  
        if "application" not in sample:  
            sample["application"] = sample.get("data_source", "unknown")

        if sample["gt_type"] == "positive":
            response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path)
        elif sample["gt_type"] == "negative":
            response = model.ground_allow_negative(instruction=sample["prompt_to_evaluate"], image=img_path)
        # print(response)
        point = response["point"]
        img_size = sample["img_size"]
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
        # Critical fix: compute correctness first
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            correctness = "wrong_format"

        # Add real-time visualization logic
        if vis_count < VISUALIZE_FIRST_N and sample["gt_type"] == "positive" and point:
            # Load image
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Create figure
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            # Draw ground-truth bbox (green)
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

            # Draw prediction (blue=correct, red=wrong)
            pred_x, pred_y = point_in_pixel
            color = 'blue' if correctness == 'correct' else 'red'
            ax.plot(pred_x, pred_y, 'o', color=color, markersize=12,
                   label=f'Prediction ({correctness})')

            # Add title
            title = f"Sample {vis_count + 1}/{VISUALIZE_FIRST_N}\n"
            title += f"App: {sample['application']} | Platform: {sample['platform']}\n"
            title += f"Instruction: {sample['prompt_to_evaluate'][:80]}...\n"
            title += f"Correctness: {correctness}"
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.axis('off')

            # Save visualization
            plt.savefig(f"{vis_dir}/sample_{vis_count:04d}_{correctness}.png",
                       bbox_inches='tight', dpi=100)
            plt.close()

            print(f"\n[Visualization] Saved sample {vis_count + 1} to {vis_dir}/")
            vis_count += 1
        # End visualization logic
        
        sample_result = {
            "id": sample["id"],
            "img_path": img_path, 
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"], 
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"], 
            "task_filename": sample["task_filename"], 
            "pred": point_in_pixel, 
            "raw_response": response["raw_response"]
        }
        
        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({
                "bbox": sample["bbox"], 
            })
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        
        sample_result.update({
            "correctness": correctness,
        })
        results.append(sample_result)
        
    result_report = evaluate(results)
    # Save to file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpot finished.")


if __name__ == "__main__":
    run_cli_aligned(parse_args())
