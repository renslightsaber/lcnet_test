import torch    
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

def Make_Optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3,                # 🔽 더 낮은 시작 learning rate
        # betas=(0.9, 0.999),     # 안정적 학습
        weight_decay=1e-4
    )

# def Make_LR_Scheduler(optimizer):
#     return CosineAnnealingLR(
#         optimizer, 
#         T_max=30,         # 전체 에폭 수
#         eta_min=1e-6      # 최저 learning rate
#     )

def Make_LR_Scheduler(optimizer):
    cosine = CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-5)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=2.0, total_epoch=8, after_scheduler=cosine
    )
    return scheduler

# Loss Function 01
# def Make_Loss_Function(number_of_classes):
#     class DiceCELoss:
#         def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
#             self.weight = weight
#             self.epsilon = epsilon
#             self.mode = mode
        
#         def __call__(self, pred, target):
#             if self.mode == 'binary':
#                 pred = pred.squeeze(1)  # shape: (batchsize, H, W)
#                 target = target.squeeze(1).float()
#                 intersection = torch.sum(pred * target, dim=(1, 2))
#                 union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
#                 dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
#                 dice_loss = 1 - dice.mean()
                
#                 ce_loss = F.binary_cross_entropy(pred, target)
            
    #         elif self.mode == 'multiclass':
    #             batchsize, num_classes, H, W = pred.shape
    #             target = target.squeeze(1)
    #             target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
    #             intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
    #             union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
    #             dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
    #             dice_loss = 1 - dice.mean()
                
    #             ce_loss = F.cross_entropy(pred, target)
    #         else:
    #             raise ValueError("mode should be 'binary' or 'multiclass'")
            
    #         combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
            
    #         return combined_loss
    
    # BINARY_SEG = True if number_of_classes==2 else False
    # return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass') 
    


def Make_Loss_Function(number_of_classes, alpha=0.3, beta=0.4, gamma=0.3, epsilon=1e-6, focal_gamma=2.0):
    class DiceCEFocalLoss:
        def __init__(self, mode='multiclass'):
            self.mode = mode
        
        def focal_loss(self, pred, target):
            if self.mode == 'binary':
                pred = pred.clamp(min=1e-6, max=1 - 1e-6)
                focal = - (target * torch.pow(1 - pred, focal_gamma) * torch.log(pred) +
                           (1 - target) * torch.pow(pred, focal_gamma) * torch.log(1 - pred))
                return focal.mean()
            elif self.mode == 'multiclass':
                logpt = F.log_softmax(pred, dim=1)
                pt = torch.exp(logpt)
                target = target.squeeze(1)
                loss = -((1 - pt) ** focal_gamma * logpt)
                return loss.gather(1, target.unsqueeze(1)).mean()
            else:
                raise ValueError("mode must be 'binary' or 'multiclass'")
        
        def __call__(self, pred, target):
            if self.mode == 'binary':
                pred = pred.squeeze(1)  # (B, H, W)
                target = target.squeeze(1).float()
                intersection = torch.sum(pred * target, dim=(1, 2))
                union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
                dice = (2 * intersection + epsilon) / (union + epsilon)
                dice_loss = 1 - dice.mean()

                bce_loss = F.binary_cross_entropy(pred, target)
                focal = self.focal_loss(pred, target)
            
            elif self.mode == 'multiclass':
                B, C, H, W = pred.shape
                target = target.squeeze(1)
                target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
                intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
                union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
                dice = (2 * intersection + epsilon) / (union + epsilon)
                dice_loss = 1 - dice.mean()

                bce_loss = F.cross_entropy(pred, target)
                focal = self.focal_loss(pred, target)
            
            else:
                raise ValueError("mode should be 'binary' or 'multiclass'")

            return alpha * bce_loss + beta * dice_loss + gamma * focal

    BINARY_SEG = True if number_of_classes == 2 else False
    return DiceCEFocalLoss(mode='binary' if BINARY_SEG else 'multiclass')