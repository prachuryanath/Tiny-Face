from tqdm import tqdm
import torch
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate

from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils import dist
from algorithm.quantize.quantized_ops_diff import ScaledLinear
# from ..utils.verification import verification
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

class ClassificationTrainer(BaseTrainer):
    def validate(self):
        if verification is None:
            logger.warning("Verification module not found. Skipping LFW validation.")
            return {'val/lfw_accuracy': 0.0} # Return dummy value

        self.model.eval()
        lfw_results = {}

        # --- Configuration for LFW (Consider moving to YAML) ---
        lfw_batch_size = configs.data_provider.get('val_batch_size', 128) # Use base_batch_size or a specific val size
        lfw_root_dir = configs.data_provider.get('lfw_root', '/path/to/lfw_data') # Add lfw_root to your YAML
        lfw_pairs_file = configs.data_provider.get('lfw_pairs_file', os.path.join(lfw_root_dir, 'lfw.bin')) # Path to lfw.bin
        image_size = (configs.data_provider.image_size, configs.data_provider.image_size)
        n_folds = configs.data_provider.get('lfw_n_folds', 10)

        if not os.path.exists(lfw_pairs_file):
            logger.warning(f"LFW pairs file not found at {lfw_pairs_file}. Skipping validation.")
            return {'val/lfw_accuracy': 0.0}

        try:
            # --- Load LFW Data (adapted from verification.load_bin) ---
            # load_bin returns a list [data_flip0, data_flip1], issame_list
            data_list, issame_list = verification.load_bin(lfw_pairs_file, image_size)
            logger.info(f"Loaded LFW data: {len(issame_list)} pairs.")
        except Exception as e:
            logger.error(f"Error loading LFW data: {e}. Skipping validation.")
            return {'val/lfw_accuracy': 0.0}


        embeddings_list = []
        with torch.no_grad():
            for data in data_list: # Process non-flipped and flipped images
                embeddings = None
                num_images = data.shape[0]
                steps = num_images // lfw_batch_size
                if num_images % lfw_batch_size != 0:
                    steps += 1

                for i in range(steps):
                    start = i * lfw_batch_size
                    end = min((i + 1) * lfw_batch_size, num_images)
                    batch_data = data[start:end]

                    # --- Preprocess and Inference ---
                    # Normalize based on TinyTraining's mean/std (likely 0.5, 0.5)
                    # Check algorithm/core/dataset/vision/transform/transform.py for mean/std
                    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda()
                    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).cuda() # Assuming TinyTraining uses 0.5/0.5
                    
                    img_tensor = batch_data.cuda().float() / 255.0 # Assuming load_bin gives 0-255
                    img_tensor = (img_tensor - mean) / std

                    # --- Get embeddings from the TinyTraining model ---
                    # The model already outputs embeddings directly
                    batch_embeddings = self.model(img_tensor).detach().cpu().numpy()

                    if embeddings is None:
                        embeddings = np.zeros((num_images, batch_embeddings.shape[1]))
                    embeddings[start:end, :] = batch_embeddings
                embeddings_list.append(embeddings)

        # --- Combine flipped and non-flipped embeddings ---
        if len(embeddings_list) == 2:
            embeddings = embeddings_list[0] + embeddings_list[1] # Add embeddings
        else: # Handle case where only non-flipped were processed (if load_bin was modified)
            embeddings = embeddings_list[0]
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # Normalize

        # --- Evaluate LFW Accuracy (adapted from verification.evaluate) ---
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        actual_issame = np.asarray(issame_list)

        tpr, fpr, accuracy = verification.calculate_roc(
            thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=n_folds
        )
        acc_mean = np.mean(accuracy)
        acc_std = np.std(accuracy)
        
        logger.info(f"LFW Validation Accuracy: {acc_mean:.5f} +/- {acc_std:.5f}")

        # --- Store results ---
        lfw_results['val/lfw_accuracy'] = acc_mean
        lfw_results['val/lfw_std'] = acc_std
        # You could add val/far calculation here too if needed, adapting from verification.calculate_val

        return lfw_results
    
    def train_one_epoch(self, epoch):
        self.model.train()
        if dist.size() > 1:
            self.data_loader['train'].sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1')

        arcface_criterion = self.arcface_criterion
        ce_criterion = self.criterion
        class_centers = self.class_centers

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            for _, (images, labels) in enumerate(self.data_loader['train']):
                # images, labels = images, labels
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()

                # --- Loss Calculation START (Adapted from arcface_torch) ---
                embeddings = self.model(images) # Get 128-dim embeddings
                # Normalize embeddings and the SEPARATE class center weights
                embeddings_normalized = F.normalize(embeddings)
                # <<<--- USE self.class_centers ---
                weights_normalized = F.normalize(class_centers)

                # --- Check shapes just before F.linear ---
                # print(f"DEBUG train: embeddings shape: {embeddings_normalized.shape}") # Should be (batch, 128)
                # print(f"DEBUG train: class_centers shape: {weights_normalized.shape}") # Should be (10000, 128)
                # print(f"DEBUG Labels min: {labels.min()}, max: {labels.max()}, unique: {torch.unique(labels)}") # Uncomment for debugging
                # Calculate logits: (batch, 128) @ (128, 10000) -> (batch, 10000)
                logits = F.linear(embeddings_normalized, weights_normalized)

                # Apply ArcFace margin
                modified_logits = arcface_criterion(logits, labels)
                # Calculate CrossEntropy loss on the *modified* logits
                loss = ce_criterion(logits, labels)
                # --- Loss Calculation END ---

                # backward and update (remains the same)
                loss.backward()

                # print("-" * 20, "Gradient Check", "-" * 20)
                # # Check gradient of class centers
                # if class_centers.grad is not None:
                #     print(f"Class Centers Grad Norm: {torch.linalg.norm(class_centers.grad).item()}")
                #     # print(f"Class Centers Grad Sample: {class_centers.grad.flatten()[0:5]}") # Optional: view some values
                # else:
                #     print("Class Centers Grad is None!")

                # Check gradient of the final embedding layer's weight (ScaledLinear)
                last_layer = self.model.module[-2] if isinstance(self.model, DistributedDataParallel) else self.model[-2]
                # if isinstance(last_layer, ScaledLinear) and last_layer.weight.grad is not None:
                #     print(f"Final Layer Weight Grad Norm: {torch.linalg.norm(last_layer.weight.grad).item()}")
                #     # print(f"Final Layer Weight Grad Sample: {last_layer.weight.grad.flatten()[0:5]}") # Optional
                # else:
                #     print("Final Layer Weight Grad is None or Layer not ScaledLinear!")

                # Check gradient of an early backbone layer (e.g., first conv)
                first_conv = self.model.module[0] if isinstance(self.model, DistributedDataParallel) else self.model[0]
                # Assuming the first layer has weights and requires grad
                # if hasattr(first_conv, 'weight') and first_conv.weight.grad is not None:
                #     print(f"First Conv Weight Grad Norm: {torch.linalg.norm(first_conv.weight.grad).item()}")
                # else:
                #     print("First Conv Weight Grad is None or layer has no weight!")
                # print("-" * 56)
                # partial update config
                if configs.backward_config.enable_backward_config:
                    from core.utils.partial_backward import apply_backward_config
                    apply_backward_config(self.model, configs.backward_config)

                if hasattr(self.optimizer, 'pre_step'):  # for SGDScale optimizer
                    self.optimizer.pre_step(self.model)
                self.optimizer.step()
                if hasattr(self.optimizer, 'post_step'):  # for SGDScaleInt optimizer
                    self.optimizer.post_step(self.model)

                # after one step
                train_loss.update(loss, images.shape[0])
                acc1 = accuracy(logits, labels, topk=(1,))[0]
                train_top1.update(acc1.item(), images.shape[0])

                t.set_postfix({
                    'loss': train_loss.avg.item(),
                    'top1': train_top1.avg.item(),
                    'batch_size': images.shape[0],
                    'img_size': images.shape[2],
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                t.update()

                # after step (NOTICE that lr changes every step instead of epoch)
                self.lr_scheduler.step()

        return {
            'train/top1': train_top1.avg.item(),
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }
