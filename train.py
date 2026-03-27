import argparse
import os
import sys
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import yaml
from albumentations import RandomRotate90, HorizontalFlip, Resize, Normalize, Compose
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from thop import profile, clever_format

from model import DeShiftNet
from losses import BCEDiceLoss
from dataset import Dataset
from utils import AverageMeter, str2bool, iou_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='experiment_name',
                        help='model name: (default: experiment_name)')
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DeShiftNet')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    
    # EMCAD ablation switches
    parser.add_argument('--use_deform_shift_block', default=True, type=str2bool,
                        help='Enable deformable Shift-MLP mixing in encoder (DeShiftNet)')
    parser.add_argument('--use_deform_tok_branch', default=True, type=str2bool,
                        help='Enable deformable token branch in MSDC stride=1 blocks')
    parser.add_argument('--deform_max_shift', default=2, type=int,
                        help='Max token shift for deformable branch')
    parser.add_argument('--use_cag', default=True, type=str2bool,
                        help='Use CAG gating instead of LGAG')
    parser.add_argument('--cag_ks', default=7, type=int,
                        help='CAG local kernel size')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        help='loss function')
    
    # dataset
    parser.add_argument('--dataset', default='your_dataset_name',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    model.train()
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit='img')
    
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda().float()
        
        # compute output
        if args.deep_supervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            output = outputs[-1]
        else:
            output = model(input)
            loss = criterion(output, target)
            
        iou, dice = iou_score(output, target)
        
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def validate(args, val_loader, model, criterion):
    model.eval()
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validation", unit='img')
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda().float()
            
            if args.deep_supervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)
            
            iou, dice = iou_score(output, target)
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main():
    args = parse_args()
    
    if args.name is None:
        if args.deep_supervision:
            args.name = '%s_%s_wDS' % (args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' % (args.dataset, args.arch)
            
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)
        
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')
    
    with open('models/%s/config.yml' % args.name, 'w') as f:
        yaml.dump(vars(args), f)
        
    # Set seed
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        
    # Define transforms
    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        Resize(args.input_h, args.input_w),
        Normalize(),
        ToTensorV2(transpose_mask=True),
    ])
    val_transform = Compose([
        Resize(args.input_h, args.input_w),
        Normalize(),
        ToTensorV2(transpose_mask=True),
    ])
    
    # Create Dataset
    # Assuming data is in data/{dataset}/images/train and data/{dataset}/images/val
    data_dir = os.path.join('data', args.dataset)
    
    # Train paths
    train_img_dir = os.path.join(data_dir, 'images', 'train')
    train_mask_dir = os.path.join(data_dir, 'masks', 'train')
    train_img_ids = glob(os.path.join(train_img_dir, '*' + args.img_ext))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    # Val paths
    val_img_dir = os.path.join(data_dir, 'images', 'val')
    val_mask_dir = os.path.join(data_dir, 'masks', 'val')
    val_img_ids = glob(os.path.join(val_img_dir, '*' + args.img_ext))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.num_classes,
        transform=train_transform)
        
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        num_classes=args.num_classes,
        transform=val_transform)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
        
    # Model
    print("Creating model: %s" % args.arch)
    model = DeShiftNet(
        num_classes=args.num_classes,
        input_channels=args.input_channels,
        channels=(32, 64, 128, 160, 256),
        use_deform_shift_mlp=args.use_deform_shift_block,
        use_cag=args.use_cag,
        cag_ks=args.cag_ks,
        use_deform_tok_branch=args.use_deform_tok_branch,
        deform_max_shift=args.deform_max_shift,
        deep_supervision=args.deep_supervision
    )
    
    model = model.cuda()
    

    # Loss
    if args.loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss().cuda()
    else:
        criterion = nn.BCEWithLogitsLoss().cuda()
        
    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        
    # Scheduler
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        milestones = [int(x) for x in args.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
        
    # Training Loop
    best_iou = 0
    trigger = 0
    
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        
        # Train
        train_log = train(args, train_loader, model, criterion, optimizer, epoch, scheduler)
        
        # Validate
        val_log = validate(args, val_loader, model, criterion)
        
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
            
        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))
              
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        
        pd.DataFrame(log).to_csv('models/%s/log.csv' % args.name, index=False)
        
        trigger += 1
        
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' % args.name)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
            
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), 'models/%s/model_epoch_%d.pth' % (args.name, epoch + 1))
            print(f"=> saved model at epoch {epoch + 1}")
            
        # Early stopping
        if args.early_stopping >= 0 and trigger >= args.early_stopping:
            print("=> early stopping")
            break
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
