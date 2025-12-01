def build_model(args):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path
    if model_type == "kimivl":
        from models.kimivl import KimiVLModel

        model = KimiVLModel()
        if model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "seeclick":
        from models.seeclick import SeeClickModel

        model = SeeClickModel()
        model.load_model()

    elif model_type == "uitars_qwen3vl_235b":  
        from models.uitars_qwen3vl_235b import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()

    elif model_type == "qwen3vl_235b":  
        from models.qwen3vl_235b import Qwen3VL235BMethod 
        model = Qwen3VL235BMethod()
    
    elif model_type == "qwen3vl_32b":  
        from models.qwen3vl_32b import Qwen3VL32BMethod
        model = Qwen3VL32BMethod()
    
    elif model_type == "qwen25vl_qwen3vl":  
            from models.qwen25vl_qwen3vl import QwenQwenComboMethod
            model = QwenQwenComboMethod()

    elif model_type == "qwen25vl_qwen3vl_235b_32b":  
                from models.qwen25vl_qwen3vl_235b_32b import Qwen25VLQwen3HybridMethod
                model = Qwen25VLQwen3HybridMethod()

    elif model_type == "qwen25vl_uitars15_7b_qwen3vl_32":  
                from models.qwen25vl_uitars15_7b_qwen3vl_32 import Qwen25VLUITarsQwen3vlTripleMethod
                model = Qwen25VLUITarsQwen3vlTripleMethod()

    elif model_type == "qwen25vl_qwen3vl_new":  
            from models.qwen25vl_qwen3vl_new import QwenQwenComboMethod
            model = QwenQwenComboMethod()

    elif model_type == "qwen3vl_32b_235b_32b":  
        from models.qwen3vl_32b_235b_32b import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()
    
    elif model_type == "qwen3vl_32b_235b_32b_mydata":  
        from models.qwen3vl_32b_235b_32b_mydata import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()

    elif model_type == "qwen3vl_32b_dual":  
        from models.qwen3vl_32b_dual import Qwen3VL32BDualMethod
        model = Qwen3VL32BDualMethod()
    
    elif model_type == "qwen3vl_32b_uitars1_5_7b_complex":  
        from models.qwen3vl_32b_uitars1_5_7b_complex import QwenUITarsDualMethod
        model = QwenUITarsDualMethod()

    elif model_type == "qwen3vl_32b_triple":  
        from models.qwen3vl_32b_triple import Qwen3VL32BTripleMethod
        model = Qwen3VL32BTripleMethod()
    
    elif model_type == "qwen3vl_32b_triple_mydata":  
        from models.qwen3vl_32b_triple_mydata import Qwen3VL32BTripleMethod
        model = Qwen3VL32BTripleMethod()

    elif model_type == "qwen3vl_235b_triple":  
            from models.qwen3vl_235b_triple import Qwen3VL235BTripleMethod
            model = Qwen3VL235BTripleMethod()

    elif model_type == "qwen3vl_235b_triple_mydata":  
                from models.qwen3vl_235b_triple_mydata import Qwen3VL235BTripleMethod
                model = Qwen3VL235BTripleMethod()
    
    elif model_type == "uitars_qwen3vl_235b_32b":  
        from models.uitars_qwen3vl_235b_32b import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()
    
    elif model_type == "uitars_qwen3vl_235b_32b_origin":  
        from models.uitars_qwen3vl_235b_32b_origin import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()
    
    elif model_type == "qwen3vl_235b_uitars_32b":  
        from models.qwen3vl_235b_uitars_32b import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()
    
    elif model_type == "qwen2_5vl":  
        from models.qwen2_5vl import Qwen2_5VL_DashscopeMethod
        model = Qwen2_5VL_DashscopeMethod()

    
    elif model_type == "qwen3vl_235b_dual":  
        from models.qwen3vl_235b_dual import Qwen3VL235BDualMethod
        model = Qwen3VL235BDualMethod()
    
    elif model_type == "qwen3vl_235b_dual_mydata":  
        from models.qwen3vl_235b_dual_mydata import Qwen3VL235BDualMethod
        model = Qwen3VL235BDualMethod()
    
    elif model_type == "qwen3vl_235b_dual_thinking":  
        from models.qwen3vl_235b_dual_thinking import Qwen3VL235BDualMethod
        model = Qwen3VL235BDualMethod()
    
    elif model_type == "qwen3vl_32b_dual_new":  
        from models.qwen3vl_32b_dual_new import UITarsQwenDualMethod
        model = UITarsQwenDualMethod()
    
    elif model_type == "qwen3vl_32b_dual_mydata":  
        from models.qwen3vl_32b_dual_mydata import Qwen3VL32BDualMethod
        model = Qwen3VL32BDualMethod()
    
    
    elif model_type == "qwen3vl_235b_triple_new":  
        from models.qwen3vl_235b_triple_new import Qwen3VL235BTripleMethod
        model = Qwen3VL235BTripleMethod()
    
    elif model_type == "qwen3vl_235b_dualnolabel":  
        from models.qwen3vl_235b_dualnolabel import Qwen3VL235BDualMethod
        model = Qwen3VL235BDualMethod()
    
    elif model_type == "qwen3vl_32b_dual":  
        from models.qwen3vl_32b_dual import Qwen3VL32BDualMethod
        model = Qwen3VL32BDualMethod()
  
    elif model_type == "qwen3vl_235b_piexl":  
            from models.qwen3vl_235b_piexl import Qwen3VL235B_PixelCoord
            model = Qwen3VL235B_PixelCoord()
    
    
    elif model_type == "qwen3vl_32b_dual_new":  
        from models.qwen3vl_32b_dual_new import Qwen3VL32BDualMethod
        model = Qwen3VL32BDualMethod()
    
    elif model_type == "qwen3vl_32b_triple_new":  
        from models.qwen3vl_32b_triple_new import Qwen3VL32BTripleMethod
        model = Qwen3VL32BTripleMethod()

    elif model_type == "qwen3vl_32b_triple_new_thinking":  
        from models.qwen3vl_32b_triple_new_thinking import Qwen3VL32BFiveLayerMethod
        model = Qwen3VL32BFiveLayerMethod()
   
    elif model_type == "qwen3vl_32b_triple_complex":  
        from models.qwen3vl_32b_triple_complex import Qwen3VL32BTripleMethod
        model = Qwen3VL32BTripleMethod()

    elif model_type == "qwen3vl_32b_dualnolabel":  
        from models.qwen3vl_32b_dualnolabel import Qwen3VL32BDualMethod
        model = Qwen3VL32BDualMethod()   
    
    elif model_type == "qwen3vl_235b_32b_dual":  
        from models.qwen3vl_235b_32b_dual import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod() 
    
    elif model_type == "qwen3vl_235b_32b_dual_mydata":  
        from models.qwen3vl_235b_32b_dual_mydata import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod() 

    elif model_type == "qwen3vl_235b_32b_dual_new":  
        from models.qwen3vl_235b_32b_dual_new import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod()
  
    elif model_type == "qwen3vl_235b_32b_235b":  
        from models.qwen3vl_235b_32b_235b import Qwen3VL235B32BDualMethod
        model = Qwen3VL235B32BDualMethod() 
    
    elif model_type == "qwenvl25_72b":  
        from models.qwenvl25_72b import Qwen25Self72b
        model = Qwen25Self72b()
    
    elif model_type == "qwen3vl_32b_dual_newnew":  
        from models.qwen3vl_32b_dual_newnew import Qwen3VL235B32BMultiCandidateMethod
        model = Qwen3VL235B32BMultiCandidateMethod()
    
    elif model_type == "qwen3vl_32b_uitars1_5_7b_32":  
            from models.qwen3vl_32b_uitars1_5_7b_32 import Qwen3VL32BTripleMethod
            model = Qwen3VL32BTripleMethod()
    
    
    elif model_type == "qwen_only":  
        from models.qwen_only import QwenSingleMethod  
        model = QwenSingleMethod() 

    elif model_type == "uitars1_5_7b_qwen3vl_32b_dual_new":  
        from models.uitars1_5_7b_qwen3vl_32b_dual_new import UITarsQwenDualMethod
        model = UITarsQwenDualMethod()
    
    elif model_type == "uitars1_5_7b_qwen3vl_235b_32b_new":  
        from models.uitars1_5_7b_qwen3vl_235b_32b_new import UITarsQwen3VLHybridMethod
        model = UITarsQwen3VLHybridMethod()

    elif model_type == "uitars1_5_7b":  
            from models.uitars1_5_7b import UITarsSingleMethod
            model = UITarsSingleMethod() 

    elif model_type == "uitars1_5_7b_dual":  
        from models.uitars1_5_7b_dual import UITarsSingleMethod
        model = UITarsSingleMethod() 
    
    elif model_type == "uitars1_5_7b_dual_mydata":  
        from models.uitars1_5_7b_dual_mydata import UITarsDualMethod
        model = UITarsDualMethod()
    
    elif model_type == "uitars1_5_7b_triple_mydata":  
        from models.uitars1_5_7b_triple_mydata import UITarsTripleMethod
        model = UITarsTripleMethod()

    # UITars + Kimi 组合方法  
    elif model_type == "uitars_kimi":  
        from models.uitars_kimivl import UITarsKimiMethod  
        model = UITarsKimiMethod()  
    elif model_type == "kimi_uitars":  
        from models.kimivl_uitars import KimiUITarsMethod  
        model = KimiUITarsMethod()

     # InternVL3 单独模型  
    elif model_type == "internvl3_only":  
        from models.internvl3_only import InternVL3OnlyMethod  
        model = InternVL3OnlyMethod()

    # Gemini 单独模型  
    elif model_type == "gemini_only":  
        from models.gemini_only import GeminiOnlyMethod  
        model = GeminiOnlyMethod()  
        
    elif model_type == "qwen1vl":
        from models.qwen1vl import Qwen1VLModel

        model = Qwen1VLModel()
        model.load_model()
    elif model_type == "qwen2vl":
        from models.qwen2vl import Qwen2VLModel

        model = Qwen2VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "qwen2_5vl":
        from models.qwen2_5vl import Qwen2_5VLModel

        model = Qwen2_5VLModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
    elif model_type == "holo1_5":
        from models.holo1_5 import Holo1_5Model

        model = Holo1_5Model()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    elif model_type == "minicpmv":
        from models.minicpmv import MiniCPMVModel

        model = MiniCPMVModel()
        model.load_model()
    elif model_type == "internvl":
        from models.internvl import InternVLModel

        model = InternVLModel()
        model.load_model()

    elif model_type in ["gpt4o", "gpt4v"]:
        from models.gpt4x import GPT4XModel

        model = GPT4XModel()

    elif model_type == "gpt5":
        from models.gpt5 import GPT5Model

        model = GPT5Model()

    elif model_type == "osatlas-4b":
        from models.osatlas4b import OSAtlas4BModel

        model = OSAtlas4BModel()
        model.load_model()
    elif model_type == "osatlas-7b":
        from models.osatlas7b import OSAtlas7BModel

        model = OSAtlas7BModel()
        model.load_model()
    elif model_type == "uground":
        from models.uground import UGroundModel

        model = UGroundModel()
        model.load_model()

    elif model_type == "fuyu":
        from models.fuyu import FuyuModel

        model = FuyuModel()
        model.load_model()
    elif model_type == "showui":
        from models.showui import ShowUIModel

        model = ShowUIModel()
        model.load_model()
    elif model_type == "ariaui":
        from models.ariaui import AriaUIVLLMModel

        model = AriaUIVLLMModel()
        model.load_model()
    elif model_type == "cogagent":
        from models.cogagent import CogAgentModel

        model = CogAgentModel()
        model.load_model()
    elif model_type == "cogagent24":
        from models.cogagent24 import CogAgent24Model

        model = CogAgent24Model()
        model.load_model()

    # Methods
    elif model_type == "screenseeker":
        from models.methods.screenseeker import ScreenSeekeRMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = ScreenSeekeRMethod(planner="gpt-4o-2024-05-13", grounder=grounder)
    elif model_type == "reground":
        from models.methods.reground import ReGroundMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = ReGroundMethod(grounder=grounder)
    elif model_type == "iterative_narrowing":
        from models.methods.iterative_narrowing import IterativeNarrowingMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = IterativeNarrowingMethod(grounder=grounder)
    elif model_type == "iterative_focusing":
        from models.methods.iterative_focusing import IterativeFocusingMethod
        from models.osatlas7b import OSAtlas7BVLLMModel

        grounder = OSAtlas7BVLLMModel()
        grounder.load_model()
        model = IterativeFocusingMethod(grounder=grounder)
    else:
        raise ValueError(f"Unsupported model type {model_type}.")
    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model
