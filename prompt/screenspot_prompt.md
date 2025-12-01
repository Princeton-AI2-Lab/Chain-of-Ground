
***************two layers prompt*******************************************************

#red circle label#
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


#complex label#
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




***************triple layers prompt*******************************************************

#second layer#
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


#third layer#
user_prompt = f"""You are a GUI automation assistant. The coordinate range is 0-1000 in both x and y.

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