dependencies = ['torch', 'torchvision', 'huggingface_hub']
import network as unet
from huggingface_hub import hf_hub_download
import torch
import os

def finetuned_lics():

    model_path = hf_hub_download(repo_id="a-data-odyssey/coastal-image-segmentation", 
                             filename="LICS_FINETUNE_26JUL24.pth",)

    return get_model(model_path)

def get_model(model_path):

   # model = unet.U_Net()

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    return model
