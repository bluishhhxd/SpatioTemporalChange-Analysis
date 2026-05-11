# ============================================================
# HIGH PERFORMANCE PAPER FIGURE GENERATION
# OPTIMIZED FOR RTX GPU + LARGE RAM
# ============================================================

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from main import (
    UNet,
    ChangeDataset,
    split_train_test_with_coverage
)

# ============================================================
# GPU OPTIMIZATION
# ============================================================

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {DEVICE}")

if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))

# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = "D:/SpatioTemporalChanges/dataset"

SAVE_DIR = "paper_figures"

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# IMPORTANT
# ============================================================

# ONLY FINAL MODELS
# ONLY REAL CHECKPOINTS

MODELS = {
    "CE":  "experiments/FinalCE/best_model.pth",
    "WCE":  "experiments/FinalWCE/model.pth",
    "FL":  "experiments/FinalFL/best_model.pth",
    "DL":  "experiments/FinalDL/best_model.pth",
    "GDL": "experiments/FinalGDL/best_model.pth",
    "JL":  "experiments/FinalJL/best_model.pth",
    "TL":  "experiments/FinalTL/best_model.pth",
    "FTL": "experiments/FinalFTL/best_model.pth",
    "WFL": "experiments/FinalWFL/best_model.pth",
}

# ============================================================
# CLASS COLORS
# ============================================================

np.random.seed(42)

CLASS_COLORS = np.random.randint(
    0,
    255,
    (36,3),
    dtype=np.uint8
)

# ============================================================
# DATASET
# ============================================================

print("Loading dataset...")

dataset = ChangeDataset(DATASET_ROOT)

train_idx, test_idx = split_train_test_with_coverage(
    dataset,
    train_ratio=0.8
)

print(f"Train Images: {len(train_idx)}")
print(f"Test Images : {len(test_idx)}")

# ============================================================
# TRANSITION CLASS ANALYSIS
# ============================================================

# VERY IMPORTANT
# Because your dataset is EXTREMELY imbalanced,
# random sampling is useless for visualization.
#
# We intelligently search for:
# - rare transition images
# - difficult samples
# - non-dominant patches
#
# Otherwise every image becomes:
# "Artificial -> Artificial"
# and paper quality dies.

# ============================================================

def decode_mask(mask):

    h,w = mask.shape

    out = np.zeros((h,w,3),dtype=np.uint8)

    for c in range(36):
        out[mask==c] = CLASS_COLORS[c]

    return out

# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path):

    model = UNet().to(DEVICE)

    state = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(state)

    model.eval()
    print("Model loaded successfully.")

    return model

# ============================================================
# RARE PATCH FINDER
# ============================================================

def find_interesting_patch():

    """
    Finds patches containing
    non-dominant transition classes.

    This is CRITICAL for your paper.
    """

    patch_size = 256
    checked_images = 0
    checked_patches = 0

    print("Searching for informative patch...")

    for idx in test_idx:

        checked_images += 1

        if checked_images % 10 == 0:
            print(f"Checked {checked_images} images...")

        region, file2006, file2012 = dataset.files[idx]

        lbl2006_path = os.path.join(
            dataset.lbl2006_root,
            region,
            file2012
        )

        lbl2012_path = os.path.join(
            dataset.lbl2012_root,
            region,
            file2012
        )

        a = dataset.read_label(lbl2006_path)
        b = dataset.read_label(lbl2012_path)

        valid = (a != 0) & (b != 0)

        y = a * 6 + b

        y[~valid] = 255

        H,W = y.shape

        for _ in range(40):

            checked_patches += 1

            if checked_patches % 100 == 0:
                print(f"Checked {checked_patches} patches...")

            i = random.randint(0,H-patch_size)
            j = random.randint(0,W-patch_size)

            patch = y[i:i+patch_size,j:j+patch_size]

            vals,counts = np.unique(
                patch[patch!=255],
                return_counts=True
            )

            if len(vals) < 3:
                continue

            dominant_ratio = counts.max() / counts.sum()

            # IMPORTANT
            # reject ultra-dominant patches

            if dominant_ratio > 0.93:
                continue

            print("\nFound informative patch.")
            print(f"Unique classes: {len(vals)}")
            print(f"Dominant ratio: {dominant_ratio:.4f}")

            img2006_path = os.path.join(
                dataset.img2006_root,
                region,
                file2006
            )

            img2012_path = os.path.join(
                dataset.img2012_root,
                region,
                file2012
            )

            img2006 = dataset.read_image(
                img2006_path,
                i,
                j,
                patch_size
            )

            img2012 = dataset.read_image(
                img2012_path,
                i,
                j,
                patch_size
            )

            x = np.concatenate(
                [img2006,img2012],
                axis=0
            ).astype(np.float32) / 255.0

            return (
                torch.tensor(x),
                torch.tensor(patch),
                vals
            )

    raise RuntimeError(
        "Could not find informative patch."
    )

# ============================================================
# QUALITATIVE PREDICTIONS
# ============================================================

def generate_prediction_figure(model_name):

    print(f"Generating predictions for {model_name}")

    model = load_model(MODELS[model_name])

    

    x,y,vals = find_interesting_patch()

    inp = x.unsqueeze(0).to(DEVICE)
    print("Running GPU inference...")
    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            enabled=(DEVICE=="cuda")
        ):

            pred = model(inp)

        pred = torch.argmax(pred,dim=1)[0]

        pred = pred.cpu().numpy()

    gt = y.numpy()

    rgb2006 = np.transpose(
        x[:3].numpy(),
        (1,2,0)
    )

    rgb2012 = np.transpose(
        x[3:].numpy(),
        (1,2,0)
    )

    gt_rgb = decode_mask(gt)

    pred_rgb = decode_mask(pred)

    fig,ax = plt.subplots(
        1,
        4,
        figsize=(18,5)
    )

    ax[0].imshow(rgb2006)
    ax[0].set_title("2006 RGB")

    ax[1].imshow(rgb2012)
    ax[1].set_title("2012 RGB")

    ax[2].imshow(gt_rgb)
    ax[2].set_title("Ground Truth")

    ax[3].imshow(pred_rgb)
    ax[3].set_title(f"{model_name} Prediction")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    print("Saving visualization...")
    plt.savefig(
        os.path.join(
            SAVE_DIR,
            f"{model_name}_prediction.png"
        ),
        dpi=700,
        bbox_inches="tight"
    )
    
    plt.close()
    print(f"{model_name} visualization saved.")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def generate_class_distribution():

    print("Generating class distribution...")

    counts = np.zeros(36,dtype=np.int64)

    # VERY IMPORTANT
    # SAMPLE CAREFULLY
    # because full sweep is huge

    chosen = random.sample(
        train_idx,
        min(120,len(train_idx))
    )

    for idx in chosen:
        if (chosen.index(idx)+1) % 10 == 0:
            print(
            f"Processed "
            f"{chosen.index(idx)+1}/{len(chosen)} images"
        )

        region, file2006, file2012 = dataset.files[idx]

        lbl2006_path = os.path.join(
            dataset.lbl2006_root,
            region,
            file2012
        )

        lbl2012_path = os.path.join(
            dataset.lbl2012_root,
            region,
            file2012
        )

        a = dataset.read_label(lbl2006_path)
        b = dataset.read_label(lbl2012_path)

        valid = (a != 0) & (b != 0)

        y = a * 6 + b

        y[~valid] = 255

        flat = y[y!=255]

        counts += np.bincount(
            flat,
            minlength=36
        )

    plt.figure(figsize=(16,5))

    plt.bar(np.arange(36),counts)

    plt.yscale("log")

    plt.xlabel("Transition Class")

    plt.ylabel("Pixel Count (log scale)")

    plt.title(
        "Extreme Pixel-Level Class Imbalance "
        "in HRSCD Dataset"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "class_distribution.png"
        ),
        dpi=700
    )
    print("Class distribution graph saved.")
    plt.close()

# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

def generate_metric_comparison():

    print("Generating metric comparison...")

    losses = [
        "CE",
        "WCE",
        "FL",
        "DL",
        "GDL",
        "JL",
        "TL",
        "FTL",
        "WFL"
    ]

    # REPLACE WITH YOUR REAL FINAL VALUES
    f1 = [
    0.53,   # CE
    0.63,   # WCE
    0.67,   # FL
    0.27,   # DL
    0.43,   # GDL
    0.27,   # JL
    0.45,   # TL
    0.29,   # FTL
    0.67    # WFL
    ]

    plt.figure(figsize=(12,5))

    plt.bar(losses,f1)

    plt.ylabel("F1 Score")

    plt.xlabel("Loss Function")

    plt.title(
        "Performance Comparison Across Loss Functions"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "metric_comparison.png"
        ),
        dpi=700
    )
    print("\nGenerating metric comparison graph...")
    plt.close()

# ============================================================
# HEATMAP
# ============================================================

def generate_heatmap():

    print("Generating REAL per-class heatmap...")

    labels = [
        "CE",
        "WCE",
        "FL",
        "DL",
        "GDL",
        "JL",
        "TL",
        "FTL",
        "WFL"
    ]

    heat = np.zeros((9,36))

    # =====================================================
    # REAL VALUES FROM YOUR EXPERIMENTS
    # =====================================================

    # CE
    heat[0,7]  = 1.0000

    # WCE
    heat[1,7]  = 0.7507
    heat[1,13] = 0.0210
    heat[1,14] = 0.9635
    heat[1,21] = 0.7885
    heat[1,35] = 0.2997

    # FL
    heat[2,7]  = 0.7297
    heat[2,13] = 0.5792

    # DL
    heat[3,14] = 1.0000

    # GDL
    heat[4,7]  = 1.0000

    # JL
    heat[5,14] = 1.0000

    # TL
    heat[6,7]  = 1.0000

    # FTL
    heat[7,14] = 1.0000

    # WFL
    heat[8,7]  = 0.9996
    heat[8,13] = 0.0027

    # =====================================================
    # TRANSPOSE
    # =====================================================
    
    heat = heat.T
    
    # =====================================================
    # COMPRESSED Y LABELS
    # =====================================================
    
    yticks = [
        0,1,2,3,4,5,6,
        7,
        8,9,10,11,12,
        13,
        14,
        15,16,17,18,19,20,
        21,
        22,
        35
    ]
    
    ytick_labels = [
        "0","1","2","3","4","5","6",
        "7",
        "8","9","10","11","12",
        "13",
        "14",
        "15","16","17","18","19","20",
        "21",
        "22-34",
        "35"
    ]
    
    # =====================================================
    # BUILD COMPRESSED MATRIX
    # =====================================================
    
    compressed_heat = []
    
    for i in yticks:
    
        if i == 22:
            # combine 22-34 into single row
            merged = np.max(
                heat[22:35],
                axis=0
            )
            compressed_heat.append(merged)
    
        else:
            compressed_heat.append(heat[i])
    
    compressed_heat = np.array(compressed_heat)
    
    # =====================================================
    # PLOT
    # =====================================================
    
    plt.figure(figsize=(6,8))
    
    sns.heatmap(
        compressed_heat,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=ytick_labels,
        annot=False
    )
    
    plt.yticks(rotation=0, fontsize=9)
    plt.xticks(rotation=0, fontsize=9)
    
    plt.xlabel("Loss Function")
    
    plt.ylabel("Transition Class")
    
    plt.title(
        "Per-Class Accuracy Across Loss Functions"
    )
    
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "per_class_heatmap_vertical.png"
        ),
        dpi=700,
        bbox_inches="tight"
    )
    
    print("Compressed vertical heatmap saved.")
    
    plt.close()

# ============================================================
# GENERATE EVERYTHING
# ============================================================

# generate_prediction_figure("CE")

# generate_prediction_figure("FL")

# generate_prediction_figure("FTL")

# generate_class_distribution()

# generate_metric_comparison()

generate_heatmap()

print("\nDONE.")
print(f"Figures saved inside: {SAVE_DIR}")