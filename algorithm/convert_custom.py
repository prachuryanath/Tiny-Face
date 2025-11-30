import os
import os.path as osp
import json
import torch
import torch.nn as nn
from compilation.convert import (
    pth_model_to_ir,
    generated_backward_graph
)
from compilation.mod import mod_save
from algorithm.core.model import build_mcu_model
from algorithm.core.utils.config import configs, load_config_from_file
from tvm import relay
from tvm.relay import ExprMutator

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
MODEL_NAME = "mcunet-5fps"
INPUT_RES = 64  # Match your training resolution
NUM_CLASSES = 10
CHECKPOINT_PATH = "algorithm/phase1_compatible.pth"
CONFIG_FILE = "algorithm/configs/config(5).yaml"
INT8_BP = False

# Sparse Update Configs (Copied from original mcu_ir_gen.py)
sparse_update_config = {
    "49kb": {
        "enable_backward_config": 1, "n_bias_update": 20, "n_weight_update": 0, "weight_update_ratio": [0, 0.25, 0.5, 0.5, 0, 0], "manual_weight_idx": [23, 24, 27, 30, 33, 39], "weight_select_criteria": "magnitude+", "pw1_weight_only": 0,
    },
    # Add others if needed
}

def build_and_load_model():
    # 1. Setup Config
    load_config_from_file(CONFIG_FILE)
    configs.net_config.net_name = MODEL_NAME
    configs.data_provider.num_classes = NUM_CLASSES
    configs.net_config.mcu_head_type = "quantized" # Force quantized structure

    print(f"Building model: {MODEL_NAME}...")
    model = build_mcu_model()

    # 2. Load Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Clean prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # --- ROBUST LOADING ---
        # The structure should match perfectly because train_cls.py saved it that way.
        # But we add strict=False to be safe against minor attribute mismatches.
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found!")
    
    return model

# --- Helpers ---
class ExtractMetaConstants(ExprMutator):
    def __init__(self):
        super().__init__()
        self.constants = []
    def visit_constant(self, const: relay.expr.Constant):
        np_data = const.data.numpy()
        if np_data.size == 1:
            new_const = relay.const(np_data.item(), dtype=str(np_data.dtype))
        else:
            new_const = relay.const(np_data)
        if "meta" in str(const): 
            self.constants.append(np_data)
        return new_const
    def extract_constants(self, func):
        expr = self.visit(func)
        return expr, self.constants

def extract_const_from_mod(mod):
    func = mod['main']
    _, consts = ExtractMetaConstants().extract_constants(func)
    return consts

# --- Main ---
if __name__ == "__main__":
    # 1. Prepare Model
    model = build_and_load_model()
    
    # 2. Paths
    output_dir = f"ir_zoos/{MODEL_NAME}_synth"
    if INT8_BP: output_dir += "_int8grad"
    os.makedirs(output_dir, exist_ok=True)
    
    fshape = [1, 3, INPUT_RES, INPUT_RES]
    fshape_str = "x".join([str(s) for s in fshape])

    # 3. Generate Forward IR
    print("Generating Forward Graph...")
    fwd_mod, real_params, scale_params, op_idx = pth_model_to_ir(
        model, 
        input_res=fshape, 
        num_classes=NUM_CLASSES
    )
    
    # Save Forward
    with open(f"{output_dir}/scale.json", "w") as fp:
        json.dump(scale_params, fp, indent=2)
    mod_save(fwd_mod, params=real_params, path=output_dir, mod_name=f"fwd-{fshape_str}.ir")
    
    # 4. Generate Backward Graphs
    for method in ["last_only", "full_bp"]:
        print(f"Generating {method} backward graph...")
        bwd_mod, bwd_names = generated_backward_graph(fwd_mod, op_idx, method=method, int8_bp=INT8_BP)
        
        meta_info = {"output_info": bwd_names}
        consts = extract_const_from_mod(bwd_mod)
        
        mod_save(bwd_mod, None, path=output_dir, mod_name=f"{method}-{fshape_str}.ir", meta=consts)
        with open(osp.join(output_dir, f"{method}-{fshape_str}.meta"), "w") as fp:
            json.dump(meta_info, fp, indent=2)

    # 5. Generate Sparse Backward Graphs
    for mem, cfg in sparse_update_config.items():
        print(f"Generating sparse graph for {mem}...")
        bwd_mod, bwd_names, sparse_meta = generated_backward_graph(
            fwd_mod, op_idx, method="sparse_bp", sparse_bp_config=cfg, int8_bp=INT8_BP
        )
        
        meta_info = {
            "output_info": bwd_names,
            "sparse_update_info": sparse_meta
        }
        consts = extract_const_from_mod(bwd_mod)
        
        mod_save(bwd_mod, None, path=output_dir, mod_name=f"sparse_bp-{mem}-{fshape_str}.ir", meta=consts)
        with open(osp.join(output_dir, f"sparse_bp-{mem}-{fshape_str}.meta"), "w") as fp:
            json.dump(meta_info, fp, indent=2)

    print(f"All graphs generated in {output_dir}")