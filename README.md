# DeShiftNet

## Repository Structure

```
DeShiftNet/
├── data/               # Dataset directory (e.g., your_dataset_name)
│   └── your_dataset_name/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── masks/
│           ├── train/
│           └── val/
├── model.py            # Complete DeShiftNet model definition and components
├── dataset.py          # PyTorch Dataset loading logic
├── losses.py           # Loss functions (e.g., BCEDiceLoss)
├── utils.py            # Utility functions (metrics computation, logging)
├── train.py            # Main training script
├── val.py              # Validation and inference evaluation script
└── requirements.txt    # Python dependencies
```

## Requirements

Ensure you have Python 3.10+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- PyTorch >= 1.10.0
- torchvision
- albumentations
- opencv-python
- scikit-learn
- thop (for FLOPs calculation, optional)

## Usage

### 1. Data Preparation

Place your dataset inside the `data/` directory following the structure shown above. The images and masks should be paired with the same filenames. Supported extensions include `.png`, `.jpg`, `.tif`, etc.

### 2. Training

To train the DeShiftNet model, run the `train.py` script. You can specify various hyperparameters and ablation configurations.

**Basic Training Command:**
```bash
python train.py --arch DeShiftNet --name experiment_name --dataset your_dataset_name --epochs 800 --batch_size 8 --lr 0.0001
```

**Key Arguments:**
- `--name`: Name of the experiment (used for saving models and logs).
- `--dataset`: Name of the folder under the `data/` directory.
- `--deep_supervision`: Enable deep supervision (default: True).
- `--use_deform_shift_block`: Enable Deformable Shift-MLP in the encoder (default: True).
- `--use_cag`: Use Content-Aware Gating (CAG) instead of LGAG (default: True).
- `--use_deform_tok_branch`: Enable deformable token branch in the decoder (default: True).

The script will automatically save the best model weights to `models/<experiment_name>/model.pth` based on the validation IoU.

### 3. Evaluation / Inference

To evaluate a trained model on a specific test set, use the `val.py` script.

**Evaluation Command:**
```bash
python val.py --name experiment_name --test_dataset test_folder_name
```
*(Where `test_folder_name` is a subdirectory inside `data/your_dataset_name/images/` and `masks/`)*

**Outputs:**
- Generates predicted masks in the `outputs/<experiment_name>/` directory.

