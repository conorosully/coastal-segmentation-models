dependencies = ['torch', 'torchvision', 'huggingface_hub']
from network import U_Net
from huggingface_hub import hf_hub_download
import torch
import os

def finetuned_lics():


    model_path = hf_hub_download(repo_id="conorosullyDS/coastal-segmentation-models", 
                                filename="LICS_FINETUNE_26JUL24.pth",)

                
    return get_model(model_path)

def get_model(model_path):

    model = U_Net()

    state_dict = torch.load(model_path, map_location=torch.device('cpu')) 
    model.load_state_dict(state_dict)
    model.eval()

    return model
