# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb


def get_backbone(p, pretrained=True):
    if p['backbone'] == 'vitB':
        from models.transformers.vit import vit_base_patch16_384
        import os, math

        backbone = vit_base_patch16_384(
            pretrained=False,
            drop_path_rate=0.15,
            img_size=p.TRAIN.SCALE
        )

        if pretrained:
            ckpt_path = os.path.join(
                torch.hub.get_dir(), "checkpoints",
                "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.pth"
            )
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"[vitB] checkpoint not found: {ckpt_path}")

            sd = torch.load(ckpt_path, map_location="cpu")

            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]

            new_sd = {}
            for k, v in sd.items():
                for pref in ("module.", "model.", "backbone.", "encoder."):
                    if k.startswith(pref):
                        k = k[len(pref):]
                new_sd[k] = v

            if "pos_embed" in new_sd:
                pe_ckpt = new_sd["pos_embed"]                                
                pe_model = backbone.pos_embed
                if pe_ckpt.shape != pe_model.shape:
                    gh_new = p.TRAIN.SCALE[0] // 16
                    gw_new = p.TRAIN.SCALE[1] // 16
                    num_extra_model = pe_model.shape[1] - (gh_new * gw_new)

                    def _infer_num_extra_and_gs(ntok: int):
                        for num_extra in (1, 2, 0):
                            ngrid = ntok - num_extra
                            if ngrid <= 0:
                                continue
                            gs = int(math.sqrt(ngrid))
                            if gs * gs == ngrid:
                                return num_extra, gs

                        gs = int(math.sqrt(max(ntok - 1, 1)))
                        return 1, gs

                    num_extra_ckpt, gs_old = _infer_num_extra_and_gs(pe_ckpt.shape[1])

                    if num_extra_ckpt >= num_extra_model:
                        tok = pe_ckpt[:, :num_extra_model]
                    else:
                        tok_part = pe_ckpt[:, :num_extra_ckpt]
                        tok_pad = pe_model[:, : (num_extra_model - num_extra_ckpt)].detach().cpu()
                        tok = torch.cat([tok_part, tok_pad], dim=1)

                    grid = pe_ckpt[:, num_extra_ckpt:]

                    grid = grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                    grid = F.interpolate(grid, size=(gh_new, gw_new), mode="bicubic", align_corners=False)
                    grid = grid.permute(0, 2, 3, 1).reshape(1, gh_new * gw_new, -1)

                    pe_new = torch.cat([tok, grid], dim=1).to(dtype=pe_model.dtype)

                    if pe_new.shape[1] != pe_model.shape[1]:
                        raise RuntimeError(f"[vitB] pos_embed token mismatch after resize: {pe_new.shape} vs {pe_model.shape}")

                    new_sd["pos_embed"] = pe_new
                    print(f"[vitB] resized pos_embed {tuple(pe_ckpt.shape)} -> {tuple(pe_new.shape)} (grid {gs_old}x{gs_old} -> {gh_new}x{gw_new})")

            msg = backbone.load_state_dict(new_sd, strict=False)
            print("[vitB] load_state_dict:", msg)

        backbone_channels = [768 for _ in range(4)]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0] // 16, p.TRAIN.SCALE[1] // 16] for _ in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT

    elif p['backbone'] == 'vitL':
        from models.transformers.vit import vit_large_patch16_384
        backbone = vit_large_patch16_384(pretrained=pretrained, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = [1024 for _ in range(4)]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT

    elif p['backbone'] == 'intern_b':
        from models.internimage import InternImage
        pretrained_url = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth'
        backbone = InternImage(
            core_op='DCNv3',
            channels=112,
            depths=[4, 4, 21, 4],
            groups=[7, 14, 28, 56],
            mlp_ratio=4.,
            drop_path_rate=0.4,
            norm_layer='LN',
            layer_scale=1.0,
            offset_scale=1.0,
            post_norm=True,
            with_cp=False,
            out_indices=(0, 1, 2, 3),
            init_cfg=dict(type='Pretrained', checkpoint=pretrained_url) if pretrained else None
        )
        if pretrained:
            backbone.init_weights()
        backbone_channels = [112, 224, 448, 896]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0] // 2 ** (i + 2), p.TRAIN.SCALE[1] // 2 ** (i + 2)] for i in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT

    elif p['backbone'] == 'intern_l':
        from models.internimage import InternImage
        pretrained_url = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth'
        backbone = InternImage(
            core_op='DCNv3',
            channels=160,
            depths=[5, 5, 22, 5],
            groups=[10, 20, 40, 80],
            mlp_ratio=4.,
            drop_path_rate=0.4,
            norm_layer='LN',
            layer_scale=1.0,
            offset_scale=2.0,
            post_norm=True,
            with_cp=False,
            out_indices=(0, 1, 2, 3),
            init_cfg=dict(type='Pretrained', checkpoint=pretrained_url) if pretrained else None
        )
        if pretrained:
            backbone.init_weights()
        backbone_channels = [160, 320, 640, 1280]
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0] // 2 ** (i + 2), p.TRAIN.SCALE[1] // 2 ** (i + 2)] for i in range(4)]
        p.final_embed_dim = p.embed_dim + p.PRED_OUT_NUM_CONSTANT

    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    if p['head'] == 'mlp':
        from models.transformers.decoder import MLPHead
        return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError


def get_model(p, pretrained=True):
    backbone, backbone_channels = get_backbone(p, pretrained=pretrained)

    if p['model'] == 'B3Net':
        from models.b3net import B3Net
        feat_channels = p.final_embed_dim
        heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
        model = B3Net(p, backbone, backbone_channels, heads)
    else:
        raise NotImplementedError('Unknown model {}. Supported model: B3Net'.format(p['model']))
    return model

def get_transformations(p):
    from data import transforms
    import torchvision

    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False)

    return database


def get_train_dataloader(p, dataset, sampler):
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False)

    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_test_dataloader(p, dataset):
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader

def get_loss(p, task=None):
    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index)

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        criterion = L1Loss()

    else:
        criterion = None

    return criterion


def get_criterion(p):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
    loss_weights = p['loss_kwargs']['loss_weights']
    return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)

def get_optimizer(p, model):
    print('Optimizer uses a single parameter group - (Default)')
    params = model.parameters()

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, params), lr=p['optimizer_kwargs']['lr'],
                                      betas=(p['optimizer_kwargs']['betas']['a'], p['optimizer_kwargs']['betas']['b']),
                                      weight_decay=p['optimizer_kwargs']['weight_decay'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    if p.scheduler == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(optimizer, p.max_iter, gamma=0.9, min_lr=0)
    elif p.scheduler == 'step':
        scheduler = torch.optim.MultiStepLR(optimizer, milestones=p.scheduler_kwargs.milestones, gamma=p.scheduler_kwargs.lr_decay_rate)

    return scheduler, optimizer
