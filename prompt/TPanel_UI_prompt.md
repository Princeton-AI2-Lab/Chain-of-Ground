
***************two layers prompt*******************************************************

#red circle label#
prompt = f"""Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

    Core Requirements:
    1. Red circle [{x_1000}, {y_1000}] (0-1000 scale) is the initial detection; only the bottom button area is valid.
    2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
    3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

    Output Format (STRICT):
    <tool_call>
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}
    </tool_call>

    x,y are integers [0,1000]. Output ONLY content within tags. Coordinates must lie in the bottom button area.""" 




***************triple layers prompt*******************************************************

#second layer#
prompt = f"""Correct red circle position: Locate "{instruction}" ONLY in the bottom button area.

    Core Requirements:
    1. Red circle [{x_1000}, {y_1000}] (0-1000 scale) is the initial detection; only the bottom button area is valid.
    2. If red circle is NOT in the bottom button area → Correct to the corresponding button in this area.
    3. If red circle IS in the bottom button area → Keep coordinates if accurate, otherwise adjust to the target button's center.

    Output Format (STRICT):
    <tool_call>
    {{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [x, y]}}}}
    </tool_call>

    x,y are integers [0,1000]. Output ONLY content within tags. Coordinates must lie in the bottom button area.""" 


#third layer#
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