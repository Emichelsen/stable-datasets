"""SimCLR: contrastive learning with 2 augmented views."""

from __future__ import annotations

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms

from benchmarks.models import (
    build_optim_config,
    collate_multiview,
    create_backbone,
    create_projector,
    get_embedding_dim,
    ssl_augmentation,
    val_transform,
)


NUM_VIEWS = 2


def create_transforms(ds_config, model_cfg=None):
    """Returns (train_transform, val_transform, collate_fn)."""
    h, w = ds_config.image_size
    if ds_config.name in {"kannadamnist", "devanagari"}:
        # Keep the full character and its orientation. Shrinking leaves room
        # for translation without clipping strokes at the image boundary.
        view = transforms.Compose(
            transforms.RGB(),
            transforms.Resize((h, w)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.02, 0.02),
                scale=(0.85, 0.95),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ToImage(mean=ds_config.mean, std=ds_config.std),
        )
    else:
        view = ssl_augmentation(ds_config, (h, w), crop_scale=(0.08, 1.0))
    train = transforms.MultiViewTransform([view] * NUM_VIEWS)
    return train, val_transform(ds_config), collate_multiview


def build(cfg, ds_config) -> tuple[spt.Module, int]:
    backbone = create_backbone(cfg.backbone, ds_config)
    embed_dim = get_embedding_dim(backbone)
    projector = create_projector(embed_dim, cfg.model.projector.hidden_dim, cfg.model.projector.output_dim)
    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr,
        simclr_loss=spt.losses.NTXEntLoss(temperature=cfg.model.loss.temperature),
        optim=build_optim_config(cfg.model),
    )
    return module, embed_dim
