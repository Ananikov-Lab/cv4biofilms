import os, sys
sys.path.insert(0, os.path.abspath('..'))
from network_training.network_training import SegModel
import torch
from tqdm import tqdm
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import pickle as pkl
from argparse import ArgumentParser

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--model_path", type=str, help="path_to_model")
    parser.add_argument("--substrate_filter", type=int, default=0, help="If 1, implement substrate filter model")
    args = parser.parse_args()
    
    CKPT_PATH = args.model_path

    arch = smp.Unet
    if args.substrate_filter:
        classes = 1
        print('Substrate Filter is implemented')
    else:
        classes = 4
        
        
    model = arch('resnet34', in_channels=1, classes=classes, activation='sigmoid')    
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)

    test_model = SegModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, batch_size=5, model=model,
                                               optimizer=opt, alpha=0.75)

    dirpath = '../data/advanced_biofilm_development_dynamics_images'
    keys = os.listdir(dirpath)
    for key in keys:
        print(f"Starting key : {key}")
        key_preds = []
        files_in_key = os.listdir(os.path.join(dirpath, key))
        for filename in tqdm(files_in_key):
            if filename.endswith('.tif'):
                image = cv2.imread(os.path.join(dirpath, key, filename))
                image = image[:-64*2]/ 255 / 255
                image = torch.Tensor([[image[:, :, 0]]])
                with torch.no_grad():
                    try:
                        prds = test_model.forward(image)
                        prds = prds.numpy()
                        key_preds.append(prds)
                    except:
                        print(image.shape)


        print(f"Finishing key : {key}")
        print("...")
        if args.substrate_filter:
            save_path = f"../data/" + \
            f"kinetic_modeling_segmentation_results/{key}_results_subs.pkl"
        else:
            save_path = f"../data/" + \
            f"kinetic_modeling_segmentation_results/{key}_results.pkl"
      
        with open(
                save_path,
                'wb') as f:
            pkl.dump(key_preds, f)
