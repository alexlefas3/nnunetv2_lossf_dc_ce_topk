"""
nnUNetTrainerDiceCETopK
=======================

A minimal nnUNetTrainer subclass that swaps the default Dice+CE loss for the
Dice+CE+TopK compound loss (dc_ce_topk_loss.py), matching the best-performing
configuration reported in:

    Orouskhani M. et al. "Intracranial aneurysm segmentation with nnU-net:
    utilizing loss functions and automated vessel extraction."
    Vessel Plus 2025;9:24. https://dx.doi.org/10.20517/2574-1209.2025.42

Installation
------------
1. Copy dc_ce_topk_loss.py into your nnU-Net source tree, e.g.:
       nnunetv2/training/loss/dc_ce_topk_loss.py
2. Copy this file into:
       nnunetv2/training/nnUNetTrainer/variants/loss/
   (create the folder if it does not exist)
3. Train as usual, pointing nnU-Net at this trainer name:
       nnUNetv2_train DATASET_ID CONFIG FOLD -tr nnUNetTrainerDiceCETopK

Notes
-----
- This trainer only overrides _build_loss(); everything else (architecture,
  data augmentation, optimizer, deep supervision handling) is inherited
  unchanged from nnUNetTrainer, so behaviour stays identical to the default
  trainer except for the loss function.
- Region-based training (label_manager.has_regions, i.e. overlapping /
  one-hot regions such as in some BraTS-style tasks) is NOT supported by
  this loss as written -- ignore_label handling assumes single-channel int
  targets, same restriction as nnU-Net's own DC_and_CE_loss. If you need
  region-based training with this loss, extend it analogously to
  DC_and_BCE_loss (sigmoid + BCEWithLogitsLoss + TopK on a per-region basis).
- Default weights below (dice=1, ce=1, topk=1) are a reasonable starting
  point; the source paper tuned/weighted these terms, so treat them as
  hyperparameters to sweep for your own dataset. A second variant class
  with adjustable weights is included below for convenience.

Optimizer / schedule overrides
-------------------------------
This trainer also reproduces the training hyperparameters reported in the
paper, which DIFFER from vanilla nnU-Net defaults:

    Paper                         nnU-Net default (nnUNetTrainer)
    -------------------------     --------------------------------
    SGD, lr = 0.01                SGD, lr = 0.01          (same)
    momentum = 0.9                momentum = 0.99         (different!)
    weight_decay = 1e-4           weight_decay = 3e-5     (different!)
    cosine annealing w/ restarts  poly LR decay           (different!)
    250 epochs                    1000 epochs             (different!)

These are implemented below via __init__ (num_epochs, weight_decay) and an
overridden configure_optimizers() (momentum, CosineAnnealingWarmRestarts).
The paper does not report the restart period (T_0) or T_mult for the cosine
schedule, so T_0=50 and T_mult=1 are used as a reasonable, documented
default -- adjust cosine_t0 / cosine_t_mult below if you have a different
value in mind or want to sweep it.
"""

import numpy as np
import torch

from nnunetv2.training.loss.dc_ce_topk_loss import DC_and_CE_and_TopK_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDiceCETopK(nnUNetTrainer):
    """
    nnUNetTrainer variant using the Dice + CE + TopK compound loss, with
    optimizer/schedule/epoch settings matching Orouskhani et al. 2025
    (SGD momentum=0.9, weight_decay=1e-4, cosine annealing w/ warm restarts,
    250 epochs). See module docstring for the full comparison table.
    """

    # Hyperparameters for the compound loss -- override in a subclass or by
    # setting them right after trainer construction if you want to sweep them.
    loss_weight_dice: float = 1.0
    loss_weight_ce: float = 1.0
    loss_weight_topk: float = 1.0
    loss_topk_k: float = 10.0  # percentage of hardest voxels used by TopK

    # Optimizer / schedule hyperparameters matching the paper's reported setup
    paper_momentum: float = 0.9
    paper_weight_decay: float = 1e-4
    paper_num_epochs: int = 250
    # Cosine annealing with warm restarts -- NOT reported explicitly by the
    # paper, these are reasonable defaults; tune to your own preference.
    cosine_t0: int = 50
    cosine_t_mult: int = 1
    cosine_eta_min: float = 0.0

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # override defaults (self.num_epochs=1000, self.weight_decay=3e-5)
        # with the paper's reported values
        self.num_epochs = self.paper_num_epochs
        self.weight_decay = self.paper_weight_decay

    def configure_optimizers(self):
        # same as nnUNetTrainer.configure_optimizers() but with momentum=0.9
        # (paper) instead of the nnU-Net default of 0.99, and a cosine
        # annealing with warm restarts schedule instead of PolyLRScheduler.
        # Note: nnUNetTrainer.on_train_epoch_start() calls
        # self.lr_scheduler.step(self.current_epoch) exactly once per epoch,
        # so T_0/T_mult below are in epoch units (e.g. T_0=50 -> first
        # restart after 50 epochs), not in optimizer-step/iteration units.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.paper_momentum,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.cosine_t0,
            T_mult=self.cosine_t_mult,
            eta_min=self.cosine_eta_min,
        )
        return optimizer, lr_scheduler

    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "nnUNetTrainerDiceCETopK does not support region-based "
                "(overlapping) labels out of the box. Extend "
                "DC_and_CE_and_TopK_loss analogously to DC_and_BCE_loss if "
                "you need this."
            )

        loss = DC_and_CE_and_TopK_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            ce_kwargs={},
            topk_kwargs={'k': self.loss_topk_k},
            weight_dice=self.loss_weight_dice,
            weight_ce=self.loss_weight_ce,
            weight_topk=self.loss_weight_topk,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerDiceCETopK_weighted(nnUNetTrainerDiceCETopK):
    """
    Same as nnUNetTrainerDiceCETopK but with an example non-uniform weighting
    (Dice emphasized) -- adjust these three numbers to reproduce whatever
    weighting you settle on after your own ablation, analogous to how the
    source paper compared several weightings of Dice/CE/TopK.
    """
    loss_weight_dice: float = 1.0
    loss_weight_ce: float = 0.5
    loss_weight_topk: float = 0.5
    loss_topk_k: float = 10.0