import os, sys
sys.path.insert(0, os.path.abspath('..'))
from network_training.network_training import SegModel
import torch
import numpy as np
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import pickle as pkl

path = '../data/final_contrast.jpg'
image = cv2.imread(path) / 255


def cut_image(image, n_rows, n_columns):
    rows = np.linspace(0, int((image.shape[0] - image.shape[0] % 64)), n_rows + 1).astype(int)
    columns = np.linspace(0, int((image.shape[1] - image.shape[1] % 64)), n_columns + 1).astype(int)
    images = []
    prev_sizes = []
    for ri in range(n_rows):
        for ci in range(n_columns):
            image_ = image[rows[ri]:rows[ri + 1], columns[ci]:columns[ci + 1]]
            prev_size = (image_.shape[1], image_.shape[0])
            dsize = (2560, 1760)
            image_ = cv2.resize(image_, dsize)
            images.append(image_)
            prev_sizes.append(prev_size)

    return images, prev_sizes


images, prev_sizes = cut_image(image, 6, 6)

with open('../data/cutted_images.pkl', 'wb') as f:
    pkl.dump(images, f)

images = [images[item] / 255 for item in range(len(images))]

CKPT_PATH = '../final_models/final_model.ckpt'

arch = smp.Unet
model = arch('resnet34', in_channels=1, classes=4, activation='sigmoid')
opt = torch.optim.Adam(model.parameters(), lr=0.0002)

test_model = SegModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, batch_size=5, model=model,
                                           optimizer=opt, alpha=0.75)

preds = []
for image in tqdm(images):
    image = torch.Tensor([[image[:, :, 0]]])
    test_model.eval()
    with torch.no_grad():
        prds = test_model.forward(image)
        prds = prds.numpy()
        preds.append(prds)

with open("../data/mapping_test.pkl", 'wb') as f:
    pkl.dump((preds, prev_sizes), f)
 