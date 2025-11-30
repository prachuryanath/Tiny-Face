# an image classification trainer
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) # Ensure we can import 'core'
# Add parent dir to find verification.py if it is in root
sys.path.append(os.path.dirname(current_dir)) 

from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from core.utils.logging import logger
from core.dataset import build_dataset
from core.optimizer import build_optimizer
from core.trainer.cls_trainer import ClassificationTrainer
from quantize.quantized_ops_diff import ScaledLinear, QuantizedConv2dDiff, QuantizedAvgPoolDiff
from quantize.quantize_helper import get_weight_scales, get_quantized_weight_and_bias

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

from core.builder.lr_scheduler import build_lr_scheduler
from core.trainer.arcface_trainer import ArcFaceTrainer
from core.utils.arcface_loss import ArcFaceLoss

try:
    import core.utils.verification as ver_module
except ImportError:
    ver_module = None

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('config', metavar='FILE', help='config file')
parser.add_argument('--run_dir', type=str, metavar='DIR', help='run directory')
parser.add_argument('--evaluate', action='store_true')


def build_config():  # separate this config requirement so that we can call main() in ray tune
    # support extra args here without setting in args
    args, unknown = parser.parse_known_args()

    load_config_from_file(args.config)
    update_config_from_args(args)
    update_config_from_unknown_args(unknown)

if ver_module is not None:
    original_test = ver_module.test

    class ScaledModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            # Scale from [-1, 1] back to [-128, 127] roughly
            return self.model(x * 127.5)

    def patched_test(data_set, backbone, batch_size, nfolds=10):
        # Wrap the backbone before passing to the original test function
        return original_test(data_set, ScaledModelWrapper(backbone), batch_size, nfolds)

    # Apply the patch
    ver_module.test = patched_test

def main():
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert configs.run_dir is not None
    os.makedirs(configs.run_dir, exist_ok=True)
    logger.init()  # dump exp config
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{configs.run_dir}".')

    # set random seed
    torch.manual_seed(configs.manual_seed)
    torch.cuda.manual_seed_all(configs.manual_seed)

    # Force Phase 1 Compatibility Settings
    configs.net_config.mcu_head_type = "fp"     
    
    logger.info(f"Building Model: {configs.net_config.net_name}")
    full_model = build_mcu_model().cuda()
    
    # 3. Extract Backbone
    backbone = None
    embed_dim = 0
    
    # Check if model has the expected structure from build_mcu_model()
    # Structure: [Conv, Blocks, Mix, AvgPool, Classifier, Flatten]
    if isinstance(full_model, nn.Sequential):
        layers = list(full_model.children())
        
        # We need to find where the classifier starts to strip it off.
        # We look for the last layer that has weights (Classifier) and is NOT the Flatten layer
        classifier_index = -1
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            # Skip Flatten
            if isinstance(layer, nn.Flatten):
                continue
            # Found the last weight layer (Classifier)
            if hasattr(layer, 'weight'):
                classifier_index = i
                # Extract dim
                if hasattr(layer, 'in_channels'):
                    embed_dim = layer.in_channels
                elif hasattr(layer, 'in_features'):
                    embed_dim = layer.in_features
                break
        
        if classifier_index == -1:
             raise ValueError("Could not locate classifier layer to determine split.")

        # Create backbone by taking everything BEFORE the classifier
        # We add our own BN and Flatten
        backbone_modules = layers[:classifier_index]
        
        # Add BN to stabilize ArcFace training on quantized backbone
        # This helps fix the "Exploding XNorm" issue
        backbone_modules.append(nn.Flatten(1))
        backbone_modules.append(nn.BatchNorm1d(embed_dim))
        
        backbone = nn.Sequential(*backbone_modules)
        
    else:
        raise ValueError(f"Unknown model structure: {type(full_model)}")
    
    backbone = backbone.cuda()
    logger.info(f"Backbone Extracted. Embedding Dim: {embed_dim}")

    if dist.size() > 1:
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[dist.local_rank()])

    # 4. Setup ArcFace Loss
    metric_loss = ArcFaceLoss(embed_dim, configs.data_provider.num_classes, s=64.0, m=0.15).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # 5. Optimizer
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    metric_params = [p for p in metric_loss.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD([
        {'params': backbone_params},
        {'params': metric_params, 'weight_decay': 0.0} 
    ], lr=configs.run_config.bs256_lr, momentum=0.9, weight_decay=4e-5)

    # 6. Data & Scheduler
    dataset = build_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=configs.data_provider.base_batch_size,
        shuffle=True,
        num_workers=configs.data_provider.n_worker,
        pin_memory=True,
        drop_last=True
    )
    lr_scheduler = build_lr_scheduler(optimizer, len(train_loader))

    # 7. Start Training
    trainer = ArcFaceTrainer(
        model=backbone,
        data_loader={'train': train_loader},
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        metric_loss=metric_loss
    )

    trainer.run_training()

    # 8. Robust Saving: Fuse BN, Merge Weights, Recalibrate
    if dist.rank() == 0:
        logger.info("Fusing BatchNorm and ArcFace weights, then saving...")
        
        if metric_loss is not None:
            # --- 1. Get ArcFace Centers ---
            centers = torch.nn.functional.normalize(metric_loss.weight.data, p=2, dim=1)
            
            # --- 2. Get BatchNorm Parameters ---
            bn_layer = None
            # Find the last BN layer we added
            for module in backbone.modules():
                if isinstance(module, nn.BatchNorm1d):
                    bn_layer = module
            
            if bn_layer is None:
                logger.info("WARNING: No BatchNorm1d found. Skipping fusion.")
                fused_weight = centers
                fused_bias = None
            else:
                logger.info(f"Fusing {bn_layer} into classifier...")
                mu = bn_layer.running_mean
                var = bn_layer.running_var
                eps = bn_layer.eps
                gamma = bn_layer.weight.data
                beta = bn_layer.bias.data
                sigma = torch.sqrt(var + eps)
                
                # Fusion Math
                scale_factor = (gamma / sigma).view(1, -1)
                fused_weight = centers * scale_factor
                
                bias_shift = beta - (mu * gamma / sigma)
                fused_bias = torch.matmul(centers, bias_shift.view(-1, 1)).squeeze()
            
            # --- 3. Locate Classifier & Adapt Shape ---
            classifier = None
            if isinstance(full_model, nn.Sequential):
                # Robust search for the last weight-bearing layer
                for layer in reversed(list(full_model.children())):
                    if hasattr(layer, 'weight') and not isinstance(layer, nn.Flatten):
                        classifier = layer
                        break
            
            if classifier is not None:
                # *** FIX: Check if Classifier is Linear (2D) or Conv2d (4D) ***
                target_shape = classifier.weight.shape
                if len(target_shape) == 4: # Conv2d: [C, E, 1, 1]
                    fused_weight = fused_weight.view(target_shape[0], target_shape[1], 1, 1)
                else: # Linear: [C, E]
                    fused_weight = fused_weight.view(target_shape[0], target_shape[1])

                # Inject Weights
                if classifier.weight.shape == fused_weight.shape:
                    classifier.weight.data = fused_weight.to(classifier.weight.device)
                    
                    if classifier.bias is not None and fused_bias is not None:
                        classifier.bias.data = fused_bias.to(classifier.bias.device)
                    elif fused_bias is not None:
                        # If classifier has no bias (common in MCUNet), we can't add the BN bias shift easily.
                        # For now, we ignore it. The impact is usually small.
                        logger.info("Note: Classifier has no bias parameter. Fused bias ignored.")

                    # --- 4. Recalibrate Scales ---
                    # Only calculate scales if the model has quantization params
                    if hasattr(classifier, 'x_scale') and hasattr(classifier, 'y_scale'):
                        logger.info("Recalibrating quantization scales...")
                        w_scales = get_weight_scales(classifier.weight.data, n_bit=8)
                        
                        # Handle broadcasting for 2D vs 4D
                        if len(target_shape) == 2:
                            # x_scale might be [1, 1] or scalar, w_scales [C, 1]
                            new_effective_scale = (classifier.x_scale * w_scales.view(-1)) / classifier.y_scale
                        else:
                            new_effective_scale = (classifier.x_scale * w_scales) / classifier.y_scale
                            
                        classifier.effective_scale.data = new_effective_scale.float()
                    
                    logger.info("Successfully fused BN + ArcFace into Classifier.")
                else:
                    logger.info(f"ERROR: Shape Mismatch! Classifier: {classifier.weight.shape}, Fused: {fused_weight.shape}")
            else:
                logger.info("ERROR: Could not locate classifier layer in full_model.")

        # --- 5. Save ---
        save_path = os.path.join(configs.run_dir, 'finetuned_full_model.pth')
        torch.save(full_model.state_dict(), save_path)
        logger.info(f"Full, fused, and re-calibrated model saved to {save_path}")        
if __name__ == '__main__':
    build_config()
    main()