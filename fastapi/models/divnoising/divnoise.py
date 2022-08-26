import io

from PIL import Image
from glob import glob
from models.divnoising.nets import lightningmodel

from models.divnoising.utils import utils

import torch


# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device


import numpy as np
#importing the os module
import os


def get_model():
    '''
    Returns a pretrained DivNoising model
    Parameters
    ----------
    noisy_input: array or list
        A stack of images.
    '''
    basedir = 'models'
    model_name = 'divnoising_mouse_skull_nuclei_demo'

    #to get the current working directory
    directory = os.getcwd()

    print(directory)

    # name = glob(basedir+"/"+model_name+'_last.ckpt')[0]
    # print(name)
    # model = lightningmodel.VAELightning.load_from_checkpoint(checkpoint_path = name)
    model = torch.hub.load(
        "pytorch/vision:v0.6.0", "deeplabv3_resnet101", pretrained=True
    )
    
    
    model.to(device)

    model.eval()

    return model

def denoise_image(model, binary_image, max_size=512):

    img = Image.open(io.BytesIO(binary_image))
    noisy_input = np.asarray(img)

    x_train_crops = utils.extract_patches(noisy_input, patch_size=512, num_patches=8)

    num_samples = 10
    export_results_path = "denoised_results"
    fraction_samples_to_export = 1
    export_mmse = True
    tta = True
    mmse_results = utils.predict_and_save(x_train_crops,model,num_samples,device,
                                    fraction_samples_to_export,export_mmse,export_results_path,tta)

    mean_psnr = compute_psnr(noisy_input, mmse_results)

    denoised_image = utils.image_from_array(noisy_input, model, device)

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(denoised_image.byte().cpu().numpy()).resize(
        noisy_input.size
    )

    return r


def compute_psnr(noisy_input, mmse_results):
    PSNRs=[]
    gt=np.mean(noisy_input[:,...],axis=0)[np.newaxis,...]

    for i in range(len(mmse_results)):
        psnr=utils.PSNR(gt[0],mmse_results[i])
        PSNRs.append(psnr)
        
    mean_psnr = np.mean(PSNRs)
    return mean_psnr