import os
import torch
import torch.nn as nn
import numpy as np
import copy
import argparse

# --- Import Project Modules ---
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file
from quantize.quantize_helper import get_weight_scales
from quantize.quantized_ops_diff import QuantizedConv2dDiff, ScaledLinear, QuantizedAvgPoolDiff, QuantizedMbBlockDiff
from core.dataset import build_dataset

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
# Update these paths to match your setup
MODEL_NAME = "mcunet-5fps"
INPUT_RES = 112
NUM_CLASSES = 10 
WEIGHTS_PATH = "runs/finetuned_backbone.pth" # Path to your saved FP32 weights
CONFIG_FILE = "configs/transfer.yaml"
def get_val(x):
    """Safely get a float value from a Tensor, NumPy array, or float."""
    if hasattr(x, 'item'):
        return x.item()
    if hasattr(x, 'numpy'):
        return x.numpy().item()
    return x

def quantize_and_assign(layer, fp_weight, fp_bias=None):
    x_scale = get_val(layer.x_scale)
    y_scale = get_val(layer.y_scale)
    
    n_bit = 8
    w_scales = get_weight_scales(fp_weight, n_bit)
    w_scales = torch.clamp(w_scales, min=1e-8)
    
    w_quant_f = fp_weight / w_scales.view(-1, 1, 1, 1)
    w_quant_f = torch.round(w_quant_f)
    w_quant_f = torch.clamp(w_quant_f, min=-127, max=127)
    layer.weight.data = w_quant_f 
    
    effective_scale = (x_scale * w_scales).float() / y_scale
    layer.effective_scale.data = effective_scale.view(-1)

    if fp_bias is not None:
        bias_scale = x_scale * w_scales.view(-1)
        b_quant_f = fp_bias / bias_scale
        b_quant_f = torch.round(b_quant_f)
        layer.bias.data = b_quant_f

def transplant_weights(skeleton_model, state_dict):
    print("  [Transplant] Starting weight transplantation...")
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    transplanted_count = 0
    for name, module in skeleton_model.named_modules():
        if isinstance(module, QuantizedConv2dDiff):
            key_weight = f"{name}.weight"
            key_bias = f"{name}.bias"
            
            if key_weight in clean_state_dict:
                fp_weight = clean_state_dict[key_weight]
                fp_bias = clean_state_dict.get(key_bias, None)
                quantize_and_assign(module, fp_weight, fp_bias)
                transplanted_count += 1
            else:
                # 4 is the classifier head in MCUNet
                if "classifier" in name or name == "4":
                    pass 
                else:
                    print(f"  [Transplant] Warning: Weight not found for {name}")

    print(f"  [Transplant] Finished. Transplanted {transplanted_count} layers.")
    return skeleton_model

def get_backbone(full_model):
    """
    Extracts backbone by STRICTLY removing the last layer (Head).
    MCUNet structure: 0:Conv, 1:Blocks, 2:FeatMix, 3:Pool, 4:Head
    We want 0, 1, 2, 3.
    """
    if isinstance(full_model, nn.Sequential):
        # Unconditionally remove the last layer (Head)
        return nn.Sequential(*list(full_model.children())[:-1])
    return full_model

def get_scale_params(model):
    """Retrieves input and output scaling parameters from the BACKBONE."""
    
    # 1. Input Scale
    first_layer = model[0]
    if not hasattr(first_layer, 'x_scale'):
        raise ValueError("First layer does not have x_scale/zero_x")
    
    input_params = {
        'scale': first_layer.x_scale,
        'zero': first_layer.zero_x
    }

    # 2. Output Scale
    # We search backwards for the first layer that has a valid y_scale (not 1.0)
    out_scale = None
    out_zero = None
    found_layer = None
    
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, (QuantizedConv2dDiff, QuantizedMbBlockDiff)):
            if hasattr(m, 'y_scale') and m.y_scale is not None:
                val = get_val(m.y_scale)
                # Valid activation scales are typically small (< 0.9)
                if val < 0.99: 
                    out_scale = m.y_scale
                    out_zero = m.zero_y if hasattr(m, 'zero_y') else 0
                    found_layer = name
                    break
    
    if out_scale is None:
        print("  [Warning] Could not find valid output scale. Using 1.0.")
        out_scale = 1.0
        out_zero = 0.0
    else:
        print(f"  [Info] Using Output Scale from layer: {found_layer}")

    output_params = {
        'scale': out_scale,
        'zero': out_zero
    }
    
    return input_params, output_params

def compare_outputs(model_fp, model_q, input_res):
    print("\n--- Running Output Comparison ---")
    model_fp.eval()
    model_q.eval()
    
    # 1. Generate BOUNDED Input [-1, 1] to avoid clipping in quantized model
    dummy_input = (torch.rand(1, 3, input_res, input_res).cuda() * 2) - 1.0
    
    # 2. Get Scaling Params
    in_params, out_params = get_scale_params(model_q)
    
    in_scale = get_val(in_params['scale'])
    in_zero = get_val(in_params['zero'])
    out_scale = get_val(out_params['scale'])
    out_zero = get_val(out_params['zero'])

    print(f"  Input Scale: {in_scale:.4f}")
    print(f"  Output Scale: {out_scale:.4f}")

    # 3. Quantize Input
    input_q = (dummy_input / in_scale) + in_zero
    input_q = torch.round(input_q).clamp(-128, 127)
    
    # 4. Forward Passes
    with torch.no_grad():
        out_fp = model_fp(dummy_input)
        out_q_raw = model_q(input_q)
        
    # 5. De-quantize Output
    out_q_dequant = (out_q_raw - out_zero) * out_scale
    
    # 6. Stats
    diff = torch.abs(out_fp - out_q_dequant)
    mse = torch.mean((out_fp - out_q_dequant)**2)
    
    print(f"  FP32 Output Mean:     {out_fp.mean().item():.4f}")
    print(f"  Quant Output (Raw):   {out_q_raw.mean().item():.4f}")
    print(f"  Quant Output (Dequant): {out_q_dequant.mean().item():.4f}")
    print(f"  MSE:                  {mse.item():.4f}")
    
    if mse.item() < 2.0: 
        print("  [SUCCESS] Quantized model tracks FP32 model closely.")
    else:
        print("  [WARNING] Discrepancy detected.")

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file {CONFIG_FILE} not found.")
        return
    load_config_from_file(CONFIG_FILE)
    configs.net_config.net_name = MODEL_NAME
    configs.data_provider.num_classes = NUM_CLASSES
    
    # Build FP32 Reference
    print(f"Building FP32 Reference Model ({MODEL_NAME})...")
    configs.net_config.mcu_head_type = "fp"
    model_fp = build_mcu_model().cuda()
    
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading FP32 weights from {WEIGHTS_PATH}")
        state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
        backbone_fp = get_backbone(model_fp)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        backbone_fp.load_state_dict(clean_state_dict, strict=False)
    else:
        print("Warning: Weights not found.")

    # Build Quantized Target
    print(f"\nBuilding Quantized Target Model ({MODEL_NAME})...")
    configs.net_config.mcu_head_type = "quantized"
    
    # Keep on CPU for transplant
    model_q = build_mcu_model() 
    model_q = transplant_weights(model_q, state_dict if os.path.exists(WEIGHTS_PATH) else model_fp.state_dict())
    
    # Move to GPU
    model_q = model_q.cuda() 

    # Compare Backbones
    # Force get_backbone to ensure Head is removed
    compare_outputs(get_backbone(model_fp), get_backbone(model_q), INPUT_RES)

if __name__ == "__main__":
    main()