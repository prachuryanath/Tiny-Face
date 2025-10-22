# an image classification trainer
import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.nn.functional as F
from core.utils import dist
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args, update_config_from_unknown_args
from core.utils.logging import logger
from core.dataset import build_dataset
from core.optimizer import build_optimizer
from core.optimizer.optimizer_entry import REGISTERED_OPTIMIZER_DICT
from core.trainer.cls_trainer import ClassificationTrainer
from core.builder.lr_scheduler import build_lr_scheduler
from losses import ArcFace
from torch.nn import Parameter
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

    # create dataset
    dataset = build_dataset()
    data_loader = dict()
    for split in dataset:
        sampler = torch.utils.data.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            seed=configs.manual_seed,
            shuffle=(split == 'train'))
        data_loader[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.data_provider.base_batch_size,
            sampler=sampler,
            num_workers=configs.data_provider.n_worker,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    # create model
    model = build_mcu_model().cuda()
    # model = build_mcu_model()

    # --- Define the Class Center Weights ---
    embedding_size = configs.loss_config.embedding_size # 128
    num_classes = configs.data_provider.num_classes   

    # Create weights ONLY for loss calculation (act as class centers)
    # Use torch.nn.Parameter so they are registered for optimization
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    class_centers_tensor = torch.empty(num_classes, embedding_size, device=device)    # Initialize weights (common initialization for ArcFace)
    torch.nn.init.xavier_uniform_(class_centers_tensor)
    # Move to GPU
    class_centers = Parameter(class_centers_tensor)    

    # --- Handle Distributed Training ---
    if dist.size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank()])  # , find_unused_parameters=True)

    # criterion = torch.nn.CrossEntropyLoss()
    # Use the ArcFace implementation from your losses.py
    arcface_criterion = ArcFace(s=64.0, margin=0.5)
    ce_criterion = torch.nn.CrossEntropyLoss()

    # --- Define Optimizer ---
    # Combine backbone parameters and the new class_centers
    params_to_optimize = [
        {"params": model.parameters()},
        {"params": [class_centers]} # <<<--- ADD class_centers HERE
    ]
    # Use build_optimizer, assuming it can handle this list structure,
    # OR initialize optimizer directly as shown in comments in previous response.
    # For simplicity, let's assume direct initialization for now:
    OptimizerClass, default_params = REGISTERED_OPTIMIZER_DICT[configs.run_config.optimizer_name]
    default_params['lr'] = configs.run_config.base_lr * dist.size()
    # Add filtering for weight decay if necessary (e.g., don't apply WD to class_centers)
    # Example: filter params_to_optimize based on names/shapes before passing to optimizer

    # optimizer = OptimizerClass(params_to_optimize, **default_params)
    print("DEBUG: Using direct SGD initialization")
    optimizer = torch.optim.SGD(
        [
            {'params': model.parameters(), 'lr': configs.run_config.base_lr * dist.size()},
            {'params': [class_centers], 'lr': configs.run_config.base_lr * dist.size()}
        ],
        momentum=0, # Corresponds to _nomom
        weight_decay=configs.run_config.weight_decay)
    # --- Make sure build_optimizer in optimizer_entry.py can handle list input OR use direct init ---
    lr_scheduler = build_lr_scheduler(optimizer, len(data_loader['train']))
    # --- Trainer Initialization ---
    trainer = ClassificationTrainer(model, data_loader, ce_criterion, optimizer, lr_scheduler)
    trainer.arcface_criterion = arcface_criterion
    # --- Pass the separate class centers to the trainer ---
    trainer.class_centers = class_centers
    # kick start training
    if configs.resume:
        trainer.resume()  # trying to resume

    if configs.backward_config.enable_backward_config:
        from core.utils.partial_backward import parsed_backward_config, prepare_model_for_backward_config, \
            get_all_conv_ops
        configs.backward_config = parsed_backward_config(configs.backward_config, model)
        prepare_model_for_backward_config(model, configs.backward_config)
        logger.info(f'Getting backward config: {configs.backward_config} \n'
                    f'Total convs {len(get_all_conv_ops(model))}')

    if configs.evaluate:
        val_info_dict = trainer.validate()
        print(val_info_dict)
        return val_info_dict  # for ray tune
    else:
        val_info_dict = trainer.run_training()
        return val_info_dict  # for ray tune


if __name__ == '__main__':
    build_config()
    main()
