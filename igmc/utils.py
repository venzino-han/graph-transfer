
import logging

def get_logger(name, path):

    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)
    
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger



from easydict import EasyDict
import yaml


def get_args_from_yaml(yaml_path):

    with open(yaml_path) as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    data_cfg = cfgs['dataset']
    model_cfg = cfgs['model']
    train_cfg = cfgs['train']

    args = EasyDict(
        {   
            'key': cfgs['key'],

            'dataset': data_cfg['name'],
            'dataset_filename': data_cfg['file_name'],
            'efeat_path': data_cfg['efeat'],
            
            # model configs
            'model_type': model_cfg['type'],
            'hop': model_cfg['hop'],
            'in_nfeats': model_cfg.get('in_nfeats'),
            'out_nfeats': model_cfg.get('out_nfeats'),
            'in_efeats': model_cfg.get('in_efeats'),
            'out_efeats': model_cfg.get('out_efeats'),
            'num_heads': model_cfg.get('num_heads'),
            'node_features': model_cfg.get('node_features'),
            'parameters': model_cfg.get('parameters'),
            
            'mpnn_type': model_cfg.get('mpnn_type'),
            'edge_dropout': model_cfg['edge_dropout'],
            'review':model_cfg.get('review'),
            'rating':model_cfg.get('rating'),
            'timestamp':model_cfg.get('timestamp'),

            'latent_dims': model_cfg.get('latent_dims'), # baseline model

            #train configs
            'device':train_cfg['device'],
            'log_dir': train_cfg['log_dir'],
            'log_interval': train_cfg.get('log_interval'),
            'train_lrs': [ float(lr) for lr in train_cfg.get('learning_rates') ],
            'train_epochs': train_cfg.get('epochs'),
            'batch_size': train_cfg['batch_size'],
            'weight_decay': train_cfg.get('weight_decay', 0),
            'lr_decay_step': train_cfg.get('lr_decay_step'),
            'lr_decay_factor': train_cfg.get('lr_decay_factor'),

        }
    )

    return args

import numpy as np
import torch as th

def evaluate(model, loader, device):
    # Evaluate RMSE
    model.eval()
    mse = 0.
    for batch in loader:
        with th.no_grad():
            preds = model(batch[0].to(device))
        labels = batch[1].to(device)
        mse += ((preds - labels) ** 2).sum().item()
    mse /= len(loader.dataset)
    return np.sqrt(mse)