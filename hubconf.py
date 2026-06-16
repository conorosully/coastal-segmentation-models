dependencies = ['torch', 'torchvision', 'huggingface_hub']

from network import U_Net
from huggingface_hub import hf_hub_download
import torch

_REPO_ID = "conorosullyDS/coastal-segmentation-models"


def _load_unet(filename, img_ch):
    model_path = hf_hub_download(repo_id=_REPO_ID, filename=filename)
    model = U_Net(img_ch=img_ch, output_ch=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── LICS (Landsat, 7 bands) ────────────────────────────────────────────────

def lics_unet():
    """U-Net trained on LICS (Landsat, 7-band). Experiment 2 baseline."""
    return _load_unet("LICS_unet_sgd.pth", img_ch=7)

def lics_unet_finetuned():
    """U-Net fine-tuned on LICS (Landsat, 7-band). Experiment 4."""
    return _load_unet("LICS_ft_sgd.pth", img_ch=7)

def lics_unet_geometric():
    """U-Net trained on LICS with geometric augmentation (Landsat, 7-band). Experiment 3."""
    return _load_unet("LICS_geometric_adam.pth", img_ch=7)

def lics_unet_no_augmentation():
    """U-Net trained on LICS without augmentation (Landsat, 7-band). Experiment 3."""
    return _load_unet("LICS_none_sgd.pth", img_ch=7)


# ── SWED (Sentinel-2, 12 bands) ───────────────────────────────────────────

def swed_unet():
    """U-Net trained on SWED (Sentinel-2, 12-band). Experiment 2 baseline."""
    return _load_unet("SWED_unet_adam.pth", img_ch=12)

def swed_unet_finetuned():
    """U-Net fine-tuned on SWED (Sentinel-2, 12-band). Experiment 4."""
    return _load_unet("SWED_ft_adam.pth", img_ch=12)

def swed_unet_geometric():
    """U-Net trained on SWED with geometric augmentation (Sentinel-2, 12-band). Experiment 3."""
    return _load_unet("SWED_geometric_adam.pth", img_ch=12)

def swed_unet_no_augmentation():
    """U-Net trained on SWED without augmentation (Sentinel-2, 12-band). Experiment 3."""
    return _load_unet("SWED_none_adam.pth", img_ch=12)


# ── SANet (Gaofen-1, 4 bands) ─────────────────────────────────────────────

def sanet_unet():
    """U-Net trained on the SANet dataset (Gaofen-1, 4-band)."""
    return _load_unet("SANet_processed_unet_sgd.pth", img_ch=4)


# ── TCUNet (Gaofen-6, 8 bands) ────────────────────────────────────────────

def tcunet_unet():
    """U-Net trained on the TCUNet dataset (Gaofen-6, 8-band)."""
    return _load_unet("TCUNet_processed_unet_sgd.pth", img_ch=8)


# ── Backwards compatibility ────────────────────────────────────────────────

def finetuned_lics():
    """Deprecated: use lics_unet_finetuned(). Kept for backwards compatibility."""
    return _load_unet("LICS_FINETUNE_26JUL24.pth", img_ch=7)

def get_model(model_path):
    """Deprecated: use a named entry point instead."""
    model = U_Net()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model
