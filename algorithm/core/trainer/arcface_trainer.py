import os
import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.basic import DistributedMetric, accuracy
from ..utils.config import configs
from ..utils import dist
from ..utils.logging import logger

# Try to import verification tools. 
# Ensure verification.py is in your python path or the root folder.
try:
    from ..utils.verification import load_bin, test
except ImportError:
    load_bin = None
    test = None

class ArcFaceTrainer(BaseTrainer):
    def __init__(self, model, data_loader, criterion, optimizer, lr_scheduler, metric_loss=None):
        super().__init__(model, data_loader, criterion, optimizer, lr_scheduler)
        self.metric_loss = metric_loss
        
        # Initialize verification sets
        self.verification_sets = {}
        self.load_verification_datasets()

    def load_verification_datasets(self):
        if load_bin is None:
            logger.info("Skipping validation dataset loading (verification.py not found).")
            return
            
        ver_targets_dir = configs.data_provider.get('verification_targets_dir')
        ver_targets = configs.data_provider.get('verification_targets', [])
        image_size = configs.data_provider.get('eval_image_size', [112, 112]) 
        
        if not ver_targets_dir:
            return

        logger.info(f"Loading verification datasets from {ver_targets_dir}...")
        for target_name in ver_targets:
            bin_path = os.path.join(ver_targets_dir, f"{target_name}.bin")
            if os.path.exists(bin_path):
                logger.info(f"Loading {target_name}...")
                data_set = load_bin(bin_path, image_size)
                self.verification_sets[target_name] = data_set
            else:
                logger.info(f"Warning: {bin_path} not found.")

    def train_one_epoch(self, epoch):
        self.model.train()
        sampler = self.data_loader['train'].sampler
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        train_loss = DistributedMetric('train_loss')
        train_top1 = DistributedMetric('train_top1') 

        with tqdm(total=len(self.data_loader['train']),
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=dist.rank() > 0 or configs.ray_tune) as t:
            
            for _, (images, labels) in enumerate(self.data_loader['train']):
                images, labels = images.cuda(), labels.cuda()
                self.optimizer.zero_grad()

                # 1. Backbone Forward: Get Embeddings
                embeddings = self.model(images)
                
                # 2. ArcFace Loss Forward: Get Logits with Margin
                thetas = self.metric_loss(embeddings, labels)
                
                # 3. Calculate Loss
                loss = self.criterion(thetas, labels)
                
                # 4. Backward
                loss.backward()
                self.optimizer.step()

                # Update metrics
                train_loss.update(loss, images.shape[0])
                acc1 = accuracy(thetas, labels, topk=(1,))[0]
                train_top1.update(acc1.item(), images.shape[0])

                t.set_postfix({
                    'loss': train_loss.avg.item(),
                    'top1': train_top1.avg.item(),
                    'lr': self.optimizer.param_groups[0]['lr'],
                })
                t.update()
                self.lr_scheduler.step()

        return {
            'train/top1': train_top1.avg.item(),
            'train/loss': train_loss.avg.item(),
            'train/lr': self.optimizer.param_groups[0]['lr'],
        }

    def validate(self):
        if dist.rank() > 0 or not self.verification_sets:
            return {'val/top1': 0.0, 'val/loss': 0.0} 

        self.model.eval()
        val_results = {}
        primary_metric = 0.0

        # Handle DDP model wrapper
        model_to_eval = self.model.module if hasattr(self.model, 'module') else self.model
        
        val_batch_size = configs.data_provider.get('val_batch_size', 32)

        for target_name, data_set in self.verification_sets.items():
            logger.info(f"Verifying on {target_name}...")
            try:
                acc1, std1, acc2, std2, xnorm, _ = test(
                    data_set, model_to_eval, val_batch_size, nfolds=10
                )
                logger.info(f"[{target_name}] Acc: {acc2:.5f} +/- {std2:.5f} | XNorm: {xnorm:.5f}")
                val_results[f'val/{target_name}_acc'] = acc2
                
                if target_name == 'lfw' or primary_metric == 0.0:
                    primary_metric = acc2
            except Exception as e:
                logger.info(f"Error validating {target_name}: {e}")
        
        val_results['val/top1'] = primary_metric # Used for checkpointing
        self.model.train()
        return val_results