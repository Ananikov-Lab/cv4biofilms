import os, sys
sys.path.insert(0, os.path.abspath('..'))
from network_training.network_training import SegModel
import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import pickle as pkl


CKPT_PATH = '../final_models/final_model.ckpt'

arch = smp.Unet
model = arch('resnet34', in_channels=1, classes=4, activation='sigmoid')
opt = torch.optim.Adam(model.parameters(), lr=0.0002)

test_model = SegModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, batch_size=5, model=model,
                                           optimizer=opt, alpha=0.75)

preds = []
for i in range(5):
    if i < 2:
        img = cv2.imread(f'../data/biofilm_formation_images/test_{i+1}.jpg')
    else:
        img = cv2.imread(f'../data/biofilm_formation_images/{i+1}.tif')
    image = img[:-64*2]/ 255 / 255
    image = torch.Tensor([[image[:, :, 0]]])
    test_model.eval()
    with torch.no_grad():
        prds = test_model.forward(image)
        prds = prds.numpy()
        preds.append(prds)

with open("../data/bacteria_img_analysis/phases_results.pkl", 'wb') as f:
    pkl.dump(preds, f)