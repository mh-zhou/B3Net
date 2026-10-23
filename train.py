# Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ["WANDB_SILENT"] = "true"

from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_test_dataset, get_train_dataloader, get_test_dataloader,\
                                get_optimizer, get_model, get_criterion
from utils.logger import Logger
from utils.train_utils import train_phase
from utils.training_checkpoint import load_training_checkpoint, restore_rng_state

from torch.utils.tensorboard import SummaryWriter
import time

import torch.distributed as dist
import datetime

parser = argparse.ArgumentParser(description='Train B3-Net')
parser.add_argument('--config_exp', required=True,
                    help='Config file for the experiment')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dataset_log', default='NYUD',
                    help='Dataset config for the experiment')

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(args, p, writers):
    start_time = time.time()
    set_seed(42)

    tb_log_dir = p.root_dir + '/tb_dir'                                                     
    p.tb_log_dir = tb_log_dir
    if args.local_rank == 0:
        train_tb_log_dir = tb_log_dir + '/train'
        test_tb_log_dir = tb_log_dir + '/test'
        mkdir_if_missing(tb_log_dir)
        mkdir_if_missing(train_tb_log_dir)
        mkdir_if_missing(test_tb_log_dir)
        tb_writer_train = SummaryWriter(train_tb_log_dir)
        writers.append(tb_writer_train)
        tb_writer_test = SummaryWriter(test_tb_log_dir)
        writers.append(tb_writer_test)

        print(f"Tensorboard dir: {tb_log_dir}")
    else:
        tb_writer_train = None
        tb_writer_test = None

    checkpoint_path = p['checkpoint'] if os.path.isfile(p['checkpoint']) else None

    model = get_model(p, pretrained=checkpoint_path is None)
    if args.local_rank == 0:
        print('Model parameters: {:,}; training precision: float32'.format(
            sum(parameter.numel() for parameter in model.parameters())))

    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    criterion = get_criterion(p).cuda()

    scheduler, optimizer = get_optimizer(p, model)

    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    train_dataloader = get_train_dataloader(p, train_dataset, train_sampler)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_test_dataloader(p, test_dataset)

    resume_state = None
    if checkpoint_path is not None:
        if args.local_rank == 0:
            print('Use checkpoint {}'.format(checkpoint_path))
        checkpoint = load_training_checkpoint(
            checkpoint_path, config=p, world_size=dist.get_world_size(),
            batches_per_epoch=len(train_dataloader))
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        iter_count = checkpoint['global_step']
        resume_state = checkpoint['resume_state']
        if checkpoint['next_batch'] == len(train_dataloader):
            start_epoch += 1
            restore_rng_state(resume_state['checkpoint_rng'])
            resume_state = None
        del checkpoint
    else:
        if args.local_rank == 0:
            print('Fresh start...')
        start_epoch = 0
        iter_count = 0

    if iter_count >= p.max_iter:
        print('Training budget already completed; no optimizer update performed.')
        return
    for epoch in range(start_epoch, p['epochs']):
        train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print('Epoch %d/%d' %(epoch+1, p['epochs']))
            print('-'*10)

        end_signal, iter_count = train_phase(p, args, train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer_train, tb_writer_test, iter_count, resume_state=resume_state)
        resume_state = None

        if end_signal:
            break

    if args.local_rank == 0:
        end_time = time.time()
        run_time = (end_time-start_time) / 3600
        print('Total running time: {} h.'.format(run_time))

def main():
    args = parser.parse_args()
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)
    dist.init_process_group(backend='nccl', init_method='env://',
                            timeout=datetime.timedelta(hours=2))
    original_stdout = sys.stdout
    logger = None
    writers = []
    wandb_run = None
    try:
        p = create_config(args.config_exp, {'run_mode': 'train'})
        if args.local_rank == 0:
            logger = Logger(os.path.join(p.output_dir, 'log_file.txt'))
            sys.stdout = logger
            print(p)
            if os.environ['WANDB_MODE'] != 'disabled':
                import wandb
                wandb_run = wandb.init(project='B3_{}'.format(args.dataset_log),
                                      name=p.version_name, sync_tensorboard=True,
                                      dir=os.path.abspath(p.output_dir))
        run(args, p, writers)
    finally:
        for writer in writers:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        sys.stdout = original_stdout
        if logger is not None:
            logger.close()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
