"""
DC_and_CE_and_TopK_loss
========================

Compound loss function combining Dice (DC), standard Cross-Entropy (CE), and
TopK Cross-Entropy (TopK) losses, as reported in:

    Orouskhani M. et al. "Intracranial aneurysm segmentation with nnU-net:
    utilizing loss functions and automated vessel extraction."
    Vessel Plus 2025;9:24. https://dx.doi.org/10.20517/2574-1209.2025.42

The paper found that a weighted combination of Dice + CE + TopK losses gave
the best overall segmentation performance on the ADAM and RENJI TOF-MRA
aneurysm datasets, slightly outperforming plain Dice + CE.

This module is not part of the official nnU-Net repository. It is written
to be a drop-in addition, following the exact conventions used in
nnunetv2/training/loss/compound_losses.py (DC_and_CE_loss, DC_and_topk_loss),
so it plugs into a standard nnUNetTrainer with minimal changes.

Usage: place this file at
    nnunetv2/training/loss/compound_losses.py   (append to the existing file)
or as a standalone module imported by a custom trainer (see
custom_trainer_dc_ce_topk.py in this same delivery).
"""

import torch
from torch import nn

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_CE_and_TopK_loss(nn.Module):
    """
    Weighted sum of Dice + CrossEntropy + TopK-CrossEntropy.

    result = weight_dice * Dice + weight_ce * CE + weight_topk * TopK

    Weights do not need to sum to 1 -- set whatever combination you want.
    Defaults (1, 1, 1) reproduce an unweighted three-way sum; the paper's
    best-performing configuration used a weighted variant, so treat
    weight_ce / weight_topk / weight_dice as tunable hyperparameters.

    :param soft_dice_kwargs: kwargs forwarded to the dice_class constructor
        (e.g. {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'ddp': ...})
    :param ce_kwargs: kwargs forwarded to RobustCrossEntropyLoss
        (e.g. {} or {'weight': class_weights})
    :param topk_kwargs: kwargs forwarded to TopKLoss
        (e.g. {'k': 10} -- k is the percentage of hardest voxels kept)
    :param weight_dice: scalar weight for the Dice term
    :param weight_ce: scalar weight for the plain CE term
    :param weight_topk: scalar weight for the TopK term
    :param ignore_label: label value to exclude from all three losses.
        Only supported for non-region-based (single-channel int) targets,
        same restriction as DC_and_CE_loss / DC_and_topk_loss.
    :param dice_class: which Dice implementation to use
        (SoftDiceLoss or MemoryEfficientSoftDiceLoss)
    """

    def __init__(self,
                 soft_dice_kwargs: dict,
                 ce_kwargs: dict,
                 topk_kwargs: dict,
                 weight_dice: float = 1.0,
                 weight_ce: float = 1.0,
                 weight_topk: float = 1.0,
                 ignore_label: int = None,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()

        # nnU-Net convention: ignore_index gets injected into every kwargs
        # dict that accepts it, mirroring DC_and_CE_loss / DC_and_topk_loss.
        ce_kwargs = dict(ce_kwargs)
        topk_kwargs = dict(topk_kwargs)
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
            topk_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_topk = weight_topk
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.topk = TopKLoss(**topk_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        target must be b, c, x, y(, z) with c=1 (standard nnU-Net int-label
        target, i.e. NOT one-hot / region-based).
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                'ignore label is not implemented for one hot encoded target '
                'variables (DC_and_CE_and_TopK_loss)'
            )
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0.0

        # RobustCrossEntropyLoss expects target with the channel dim squeezed
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0.0

        # TopKLoss slices target[:, 0] internally -> pass target as-is
        topk_loss = self.topk(net_output, target) \
            if self.weight_topk != 0 and (self.ignore_label is None or num_fg > 0) else 0.0

        result = (self.weight_dice * dc_loss +
                  self.weight_ce * ce_loss +
                  self.weight_topk * topk_loss)
        return result
