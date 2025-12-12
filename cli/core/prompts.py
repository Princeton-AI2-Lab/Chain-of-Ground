INITIAL_QWEN = """**CRITICAL: You MUST output in the exact format specified. Do NOT provide explanations.**\n\nTask: {instruction}\n\nOutput ONLY:\n<tool_call>\n{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}\n</tool_call>\n\nWhere x,y are integers in [0, 1000] range."""

INITIAL_UITARS = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.\n\n## Output Format\n\nAction: ...\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction\n{instruction}"""

REFINE_UITARS_CIRCLE = """**CRITICAL: Search WITHIN the large red circle area.**

Task: {instruction}

The screenshot shows a RED TRANSPARENT CIRCLE — this is the initial detection reference.

Your job:
1. Independently verify the target element for "{instruction}"
2. If red circle position is inaccurate → Correct to the true center of the target element
3. If red circle IS accurate → Keep coordinates or adjust to the target element's precise center

Output Format (STRICT):
Action: click(point='<point>x1 y1</point>')

NO explanations. Output ONLY the action line with pixel coordinates."""

REFINE_UITARS_CIRCLE_TPanel_UI = """Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

The screenshot shows a RED TRANSPARENT CIRCLE — this is the initial detection reference; only the bottom button area is valid.

Your job:
1. Independently verify the target element for "{instruction}"
2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

Output Format (STRICT):
Action: click(point='<point>x1 y1</point>')

NO explanations. Output ONLY the action line with pixel coordinates."""


REFINE_QWEN_CIRCLE = """**CRITICAL: Search WITHIN the large red circle area.**

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

REFINE_QWEN_CIRCLE_TPanel_UI = """Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

Core Requirements:
1. Red circle [{x_1000}, {y_1000}] (0-1000 scale) is the initial detection; only the bottom button area is valid.
2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

Output Format (STRICT):
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}
</tool_call>

x,y are integers [0,1000]. Output ONLY content within tags. Coordinates must lie in the bottom button area."""


FINAL_QWEN_COMPARE = """FINAL VALIDATION: Locate "{instruction}" ONLY in the bottom button area\n\nYou see two marked points:\n- RED CIRCLE: [{first_x_1000}, {first_y_1000}] (initial detection)\n- BLUE CIRCLE: [{second_x_1000}, {second_y_1000}] (refined result)\n\nRules:\n1. Only the bottom button area is valid\n2. Choose the circle accurately centered on the target (must be in valid area)\n3. If both are incorrect/off-area, find the target's true center in the bottom button area\n4. Coordinates: integers [0,1000]\n\nOutput ONLY:\n<tool_call>\n{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}\n</tool_call>\n\nx,y integers [0,1000]."""

FINAL_QWEN_BLUEBOX = """You are a GUI automation assistant. The coordinate range is 0-1000 in both x and y.

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
