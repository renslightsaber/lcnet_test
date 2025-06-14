# 🔬 LCNet-Improved for Binary Segmentation
: This repo is based on the model **'LCNet'** proposed in [Lightweight Context-Aware Network Using Partial-Channel Transformation for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/10411824), and tried some modifications for more robust, better performance at various domain datasets. This repo is created for CAU Final Competition Project. 

Paper: [Lightweight Context-Aware Network Using Partial-Channel Transformation for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/10411824)     
Official Github: https://github.com/lztjy/LCNet     

> **Competition Training Repo**  


## 🚀 Overview

This repository contains an **improved implementation of LCNet** designed for binary segmentation tasks across diverse domains such as medical imaging, plant leaf segmentation, cracks, and car damage detection.

To start training, simply run:

```python
📁 competition_main_lcnet_improved.ipynb
```
## 🔧 Key Modifications

### 1. 📈 Dilation Strategy in LCNet

- The `dilation_block_2` configuration was modified for better multi-scale receptive field capture:

```python
dilation_block_2 = [2, 4, 8, 16, 20, 24, 32]
```

### 2. 🧠 Attention-Enhanced Fusion via SEBlock

The DAD module, originally using CBAM, has been replaced with SEBlock for lightweight yet effective channel-wise attention.

```python
# F_rg = self.cbam(F_rg)  # Added and applied CBAM Module, but 🔄 Replaced with:
F_rg = self.attn(F_rg)  # SEBlock with reduction=8. Better Performance
```
The SEBlock performs global average pooling followed by squeeze-and-excitation with a reduction ratio of 8.


## ⚙️ Training Settings
Batch Size is fixed as 16 and also number of epochs is fixed as 30.

### 🛠️ Environment

- OS: Ubuntu 20.04    
- Python: 3.10    
- torch: 2.6.0    
- torchvision: 0.21.0     
- GPU: NVIDIA RTX A6000 (x1)    

> All training and evaluation were performed on a single NVIDIA RTX A6000 GPU.

### 🧪 Loss Function

```python
TotalLoss = α * BCE + β * Dice + γ * Focal
(α=0.3, β=0.4, γ=0.3, focal_gamma=2.0)
```

### ✅ Optimizer

```python
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### 📉 Learning Rate Scheduler
```python
scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=2.0,             # 📈 warm-up to 2x LR
    total_epoch=8,              # 🔁 over 8 epochs
    after_scheduler=CosineAnnealingLR(..., T_max=30, eta_min=5e-5)
)
```

## 📊 Performance Comparison (LCNet vs Improved)

| Dataset | Method           | Loss    | IoU    | Dice   | Precision | Recall  |
|---------|------------------|---------|--------|--------|-----------|---------|
| **ETIS**   | 🔷 LCNet (original) | 0.6579 | 0.4055 | 0.5255 | 0.4487    | 0.6873  |
|           | 🟧 Improved         | 0.5037 | 0.4022 | 0.4997 | 0.4899    | 0.5673  |
| **CVPPP**  | 🔷 LCNet (original) | 0.4021 | 0.8446 | 0.9124 | 0.8491    | 0.9939  |
|           | 🟧 Improved         | 0.2931 | 0.8469 | 0.9138 | 0.8515    | 0.9940  |
| **CFD**    | 🔷 LCNet (original) | 0.7537 | 0.0911 | 0.1648 | 0.1001    | 0.6003  |
|           | 🟧 Improved         | 0.5746 | 0.2046 | 0.3368 | 0.2296    | 0.7251  |
| **CarDD**  | 🔷 LCNet (original) | 0.5736 | 0.3521 | 0.4765 | 0.5504    | 0.5357  |
|           | 🟧 Improved         | 0.4328 | 0.3917 | 0.5150 | 0.5621    | 0.5751  |

---

🔷 **LCNet (original)** : Original architecture  
🟧 **Improved** : LCNet with SEBlock + Updated Dilation Strategy




## 🧾 License & Contact

This repository is part of an academic competition submission.
For further questions, feel free to open an issue or contact the author.

⸻

## 🎯 Designed for robust generalization across multiple domains using LCNet-Improved.

