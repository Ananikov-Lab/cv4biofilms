import os, sys
sys.path.insert(0, os.path.abspath('..'))
from network_training.network_training import SegModel
import torch
from tqdm import tqdm
import numpy as np
import cv2
import time
import segmentation_models_pytorch as smp
import pickle as pkl
from argparse import ArgumentParser

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument("--model_path", type=str, help="path_to_model")
    args = parser.parse_args()
    
    CKPT_PATH = args.model_path

    arch = smp.Unet
    classes = 4
        
    model = arch('resnet34', in_channels=1, classes=classes, activation='sigmoid')    
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)

    test_model = SegModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, batch_size=5, model=model,
                                               optimizer=opt, alpha=0.75)

    dirpath = '../data/antibiotics_images'
    keys = os.listdir(dirpath)
    for key in keys:
        print(f"Starting key : {key}")
        key_times = []
        key_preds = []
        files_in_key = os.listdir(os.path.join(dirpath, key))
        for filename in tqdm(files_in_key):
            if filename.endswith('.tif'):
                
                image = cv2.imread(os.path.join(dirpath, key, filename))
                image = image[:-64*2]/ 255 / 255
                image = torch.Tensor([[image[:, :, 0]]])
                with torch.no_grad():
                    try:
                        s1 = time.time()
                        prds = test_model.forward(image)
                        s2 = time.time()
                        prds = prds.numpy()
                        key_preds.append(prds)
                        key_times.append(s2-s1)
                    except:
                        print(image.shape)


        print(f"Finishing key : {key}")
        print("...")
       
        t_save_path = f"../data/antibiotics_segmentation_results/times/{key}_times.pkl"
      
        with open(
                t_save_path,
                'wb') as f:
            pkl.dump(key_times, f)
            
        res_save_path = f"../data/antibiotics_segmentation_results/{key}_results.pkl"
      
        with open(
                res_save_path,
                'wb') as f:
            pkl.dump(key_preds, f)
            