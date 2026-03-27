import argparse
import os
import re
import time
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import DeShiftNet
from dataset import Dataset
from utils import AverageMeter, iou_score, calculate_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='experiment_name', help='model name')
    parser.add_argument('--test_dataset', default='val', help='test subfolder name under data/<dataset>/images and masks')
    # Deep supervision evaluation branch selection and weight (default last)
    parser.add_argument('--ds_branch', default='last', choices=['last','avg','weighted'], help='Evaluation fusion method for multi-head output')
    parser.add_argument('--ds_metric_weights', default='0.1,0.2,0.2,0.5', type=str, help='Weights for each head during weighted fusion, in order of [p4,p3,p2,p1]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    config_path = f'models/{args.name}/config.yml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # Create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'DeShiftNet':
        model = DeShiftNet(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            use_deform_shift_mlp=config.get('use_deform_shift_block', True),
            use_cag=config.get('use_cag', True),
            cag_ks=config.get('cag_ks', 7),
            use_deform_tok_branch=config.get('use_deform_tok_branch', True),
            deform_max_shift=config.get('deform_max_shift', 2),
            deep_supervision=config.get('deep_supervision', True)
        )
    else:
        raise ValueError(f"Unsupported architecture: {config['arch']}")
    
    model = model.cuda()

    # Prepare dataset
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
        ToTensorV2(transpose_mask=True),
    ])

    # Select test subdirectory based on parameters
    test_img_dir = os.path.join('data', config['dataset'], 'images', args.test_dataset)
    test_mask_dir = os.path.join('data', config['dataset'], 'masks', args.test_dataset)
    
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")
        
    img_ids = glob(os.path.join(test_img_dir, '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_dataset = Dataset(
        img_ids=img_ids,
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # Get all model checkpoints
    model_dir = 'models/%s' % config['name']
    model_paths = glob(os.path.join(model_dir, '*.pth'))
    
    if not model_paths:
        print(f"No models found in {model_dir}")
        return

    # Sort models by epoch number (if multiple checkpoints exist)
    def extract_epoch(path):
        s = re.findall(r'\d+', os.path.basename(path))
        return int(s[-1]) if s else -1
    model_paths.sort(key=extract_epoch)

    all_results = []
    per_image_records = []  # Per-image metrics

    for model_path in model_paths:
        print(f"\n--- Testing {os.path.basename(model_path)} ---")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue
        
        model.eval()

        iou_avg_meter = AverageMeter()
        dice_avg_meter = AverageMeter()
        precision_avg_meter = AverageMeter()
        recall_avg_meter = AverageMeter()
        f1_avg_meter = AverageMeter()
        accuracy_avg_meter = AverageMeter()
        
        # Performance metrics
        gpu_mem_meter = AverageMeter()
        inference_time_meter = AverageMeter()

        output_dir_model = os.path.join('outputs', config['name'], os.path.splitext(os.path.basename(model_path))[0])
        for c in range(config['num_classes']):
            os.makedirs(os.path.join(output_dir_model, str(c)), exist_ok=True)

        with torch.no_grad():
            for input, target, meta in tqdm(val_loader, total=len(val_loader)):
                input = input.cuda()
                target = target.cuda().float()
                
                # Reset max memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Measure inference time
                torch.cuda.synchronize()
                start_time = time.time()

                # Compatible with deep_supervision: select output based on ds_branch
                if config['deep_supervision']:
                    outputs = model(input)
                    if args.ds_branch == 'last':
                        output = outputs[-1]
                    elif args.ds_branch == 'avg':
                        output = torch.stack(outputs, dim=0).mean(dim=0)
                    elif args.ds_branch == 'weighted':
                        try:
                            weights = [float(w) for w in args.ds_metric_weights.split(',')]
                        except Exception:
                            weights = [0.1, 0.2, 0.2, 0.5]
                        if len(weights) >= len(outputs):
                            weights = weights[:len(outputs)]
                        else:
                            remain = len(outputs) - len(weights)
                            weights += [1.0] * remain
                        s = sum(weights)
                        if s > 0:
                            weights = [w / s for w in weights]
                        stacked = torch.stack(outputs, dim=0)
                        w = torch.tensor(weights, device=stacked.device).view(len(weights), 1, 1, 1)
                        output = (stacked * w).sum(dim=0)
                    else:
                        output = outputs[-1]
                else:
                    output = model(input)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                # Calculate performance metrics
                batch_size = input.size(0)
                inference_time = end_time - start_time
                inference_time_meter.update(inference_time, batch_size)
                
                max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
                gpu_mem_meter.update(max_mem, batch_size)

                # Batch-level metrics
                iou, dice = iou_score(output, target)
                iou_avg_meter.update(iou, input.size(0))
                dice_avg_meter.update(dice, input.size(0))

                output_binary = torch.sigmoid(output).cpu().numpy()
                output_binary[output_binary >= 0.5] = 1
                output_binary[output_binary < 0.5] = 0
                target_np = target.detach().cpu().numpy()
                target_np = (target_np >= 0.5).astype(int)

                precision, recall, f1, accuracy = calculate_metrics(output, target)
                precision_avg_meter.update(precision, input.size(0))
                recall_avg_meter.update(recall, input.size(0))
                f1_avg_meter.update(f1, input.size(0))
                accuracy_avg_meter.update(accuracy, input.size(0))

                # Save predicted masks and per-image metrics.
                for i in range(len(output_binary)):
                    # Save masks
                    for c in range(config['num_classes']):
                        cv2.imwrite(os.path.join(output_dir_model, str(c), meta['img_id'][i] + '.png'),
                                    (output_binary[i, c] * 255).astype('uint8'))
                    
                    # Per-image metrics (based on a single sample)
                    iou_i, dice_i = iou_score(output[i:i+1], target[i:i+1])
                    prec_i, rec_i, f1_i, acc_i = calculate_metrics(output[i:i+1], target[i:i+1])
                    per_image_records.append({
                        'checkpoint': os.path.basename(model_path),
                        'test_dataset': args.test_dataset,
                        'img_id': meta['img_id'][i],
                        'IoU': float(iou_i),
                        'Dice': float(dice_i),
                        'Precision': float(prec_i),
                        'Recall': float(rec_i),
                        'F1 Score': float(f1_i),
                        'Accuracy': float(acc_i),
                    })
        
        # Calculate FPS
        avg_inference_time = inference_time_meter.sum / inference_time_meter.count  # Time per image
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        print('IoU: %.4f' % iou_avg_meter.avg)
        print('Dice: %.4f' % dice_avg_meter.avg)
        print('Precision: %.4f' % precision_avg_meter.avg)
        print('Recall: %.4f' % recall_avg_meter.avg)
        print('F1 Score: %.4f' % f1_avg_meter.avg)
        print('Accuracy: %.4f' % accuracy_avg_meter.avg)
        print('FPS: %.2f' % fps)
        print('GPU Memory: %.2f MB' % gpu_mem_meter.avg)
        
        all_results.append({
            'model': os.path.basename(model_path),
            'IoU': iou_avg_meter.avg,
            'Dice': dice_avg_meter.avg,
            'Precision': precision_avg_meter.avg,
            'Recall': recall_avg_meter.avg,
            'F1 Score': f1_avg_meter.avg,
            'Accuracy': accuracy_avg_meter.avg,
            'FPS': fps,
            'GPU_Memory_MB': gpu_mem_meter.avg
        })

    # Save metrics to CSV and plot
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Name output based on test subdirectory
        base_dir = os.path.join('outputs', config['name'])
        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, f"{args.test_dataset}_metrics_summary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nMetrics summary saved to {csv_path}")

        # Export per-image metrics to CSV
        if per_image_records:
            per_img_df = pd.DataFrame(per_image_records)
            per_img_csv = os.path.join(base_dir, f"{args.test_dataset}_per_image_metrics.csv")
            per_img_df.to_csv(per_img_csv, index=False)
            print(f"Per-image metrics saved to {per_img_csv}")

        # Plotting
        try:
            plt.style.use('ggplot')
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(24, 18))  # Changed to 3 rows
            fig.suptitle('Model Performance Metrics', fontsize=20)
            
            metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'FPS', 'GPU_Memory_MB']
            model_names = results_df['model']

            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i//3, i%3]
                ax.plot(model_names, results_df[metric], marker='o', linestyle='-')
                ax.set_title(metric, fontsize=14)
                ax.set_xlabel('Model', fontsize=12)
                ax.set_ylabel('Score', fontsize=12)
                ax.tick_params(axis='x', labelrotation=45, labelsize=10)
                for label in ax.get_xticklabels():
                    label.set_ha('right')
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(base_dir, f"{args.test_dataset}_metrics_summary.png")
            plt.savefig(plot_path)
            print(f"Metrics plot saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate plot. Error: {e}")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
