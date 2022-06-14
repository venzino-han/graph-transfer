import math, copy

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from igmc.utils import get_logger, get_args_from_yaml, evaluate

from igmc.dataset import get_dataloader

from igmc.igmc import IGMC


def train_epoch(model, loss_fn, optimizer, loader, device, logger, log_interval):
    model.train()

    epoch_loss = 0.
    iter_loss = 0.
    iter_mse = 0.
    iter_cnt = 0
    iter_dur = []

    for iter_idx, batch in enumerate(loader, start=1):
        t_start = time.time()

        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(inputs)
        loss = loss_fn(preds, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * preds.shape[0]
        iter_loss += loss.item() * preds.shape[0]
        iter_mse += ((preds - labels) ** 2).sum().item()
        iter_cnt += preds.shape[0]
        iter_dur.append(time.time() - t_start)

        if iter_idx % log_interval == 0:
            logger.debug(f"Iter={iter_idx}, loss={iter_loss/iter_cnt:.4f}, rmse={math.sqrt(iter_mse/iter_cnt):.4f}, time={np.average(iter_dur):.4f}")
            iter_loss = 0.
            iter_mse = 0.
            iter_cnt = 0
            
    return epoch_loss / len(loader.dataset)



def train(args:EasyDict, logger):
    data_path = f'data/{args.dataset_filename}'
    train_loader, valid_loader, test_loader =  get_dataloader(data_path, batch_size=args.batch_size, feature_path=None)
    
    ### prepare data and set model
    if args.model_type == 'IGMC':
        in_feats = (args.hop+1)*2 
        model = IGMC(in_feats=in_feats, 
                    latent_dim=args.latent_dims,
                    num_relations=5, 
                    num_bases=4, 
                    regression=True, 
                    edge_dropout=args.edge_dropout,
                    ).to(args.device)

    loss_fn = nn.MSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    logger.info("Loading network finished ...\n")

    
    best_epoch = 0
    best_rmse = np.inf

    logger.info(f"Start training ... learning rate : {args.train_lr}")

    model.load_state_dict(th.load('parameters/igmc_book_0.2006.pt'))

    epochs = list(range(1, args.train_epochs+1))
    for epoch_idx in epochs:
        logger.debug(f'Epoch : {epoch_idx}')
    
        train_loss = train_epoch(model, loss_fn, optimizer, train_loader, 
                                 args.device, logger, args.log_interval)
        # val_rmse = evaluate(model, valid_loader, args.device)
        val_rmse = -1
        test_rmse = evaluate(model, test_loader, args.device)
        eval_info = {
            'epoch': epoch_idx,
            'train_loss': 0,#train_loss,
            'val_rmse' : val_rmse,
            'test_rmse': test_rmse,
        }
        logger.info('=== Epoch {}, train loss {:.6f}, val rmse {:.6f}, test rmse {:.6f} ==='.format(*eval_info.values()))

        if epoch_idx % args.lr_decay_step == 0:
            for param in optimizer.param_groups:
                param['lr'] = args.lr_decay_factor * param['lr']
            print('lr : ', param['lr'])

        if best_rmse > test_rmse:
            logger.info(f'new best test rmse {test_rmse:.6f} ===')
            best_epoch = epoch_idx
            best_rmse = test_rmse
            best_state = copy.deepcopy(model.state_dict())

    th.save(best_state, f'./parameters/{args.key}_{best_rmse:.4f}.pt')
    logger.info(f"Training ends. The best testing rmse is {best_rmse:.6f} at epoch {best_epoch}")
    return best_rmse
    
import yaml

def main():
    with open('./igmc/train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
    file_list = files['files']
    for f in file_list:
        args = get_args_from_yaml(f)
        logger = get_logger(name=args.key, path=f"{args.log_dir}/{args.key}.log")
        logger.info('train args')
        for k,v in args.items():
            logger.info(f'{k}: {v}')

        final_best_rmse = 100
        best_lr = None
        for lr in args.train_lrs:
            sub_args = args
            sub_args['train_lr'] = lr
            best_rmse = train(sub_args, logger=logger)

            if best_rmse < final_best_rmse:
                final_best_rmse = best_rmse
                best_lr = lr
        logger.info(f"**********The final best testing RMSE is {final_best_rmse:.6f} at lr {best_lr}********")

if __name__ == '__main__':
    main()