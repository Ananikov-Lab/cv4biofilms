import os
from train_network import SegModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp
import cv2
import pickle as pkl
import pytorch_lightning as pl
from network_training.dataset_preparation import SegmentedImagesMultiLabel, get_loader_multilabel
from network_training.augmentations import get_training_augmentation, get_validation_augmentation
from argparse import ArgumentParser


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/kskozlov/files_for_research/bacteria_img_analysis/cell_data_3A.pkl")
    parser.add_argument("--split", type=str, default="val",
                        help="Get examples of train or validation")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--report_images_output_dir", type=str,
                        default="/home/kskozlov/files_for_research/bacteria_img_analysis/validation_sample_model_preds")
    parser.add_argument('--metrics_dict_list_save_path', type=str, default=None,
                        help="If not None, metrics_dict_list is saved")
    parser.add_argument('--valid_preds_save_path', type=str, default=None,
                        help="If not None, valid_preds are saved")

    args = parser.parse_args()
    os.makedirs(args.report_images_output_dir, exist_ok=True)
    # Load images from pickle file
    images = []
    masks = []
    with open(args.data_path, 'rb') as f:
        for item in pkl.load(f):
            if not item:
                continue

            if any([n in item['name'] for n in ['103', '020', '018']]):
                continue

            images.append(item['image'][:-140])
            masks.append(item['mask'][:-140])

    indexes = np.arange(len(images))

    # Split image dataset and get split, which you specified in argument --split
    train, test = train_test_split(indexes, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    if args.split == 'val':
        test_images = np.array([images[item] for item in val])
        test_masks = np.array([masks[item] for item in val])
    elif args.split == 'test':
        test_images = np.array([images[item] for item in test])
        test_masks = np.array([masks[item] for item in test])
    else:
        raise ValueError('Incorrect split')

    # Place a dataset in a dataloader
    test_loader = get_loader_multilabel(SegmentedImagesMultiLabel, test_images,
                                        test_masks, 4, np.arange(len(test_images)),
                                        get_validation_augmentation,
                                        batch_size=1, shuffle=False, num_workers=4)

    # Load the model (hyperparameters used were optimized during multiple training procedures)
    CKPT_PATH = args.checkpoint_path

    arch = smp.Unet
    model = arch('resnet34', in_channels=1, classes=4, activation='sigmoid')
    opt = torch.optim.Adam(model.parameters(), lr=0.0002)

    test_model = SegModel.load_from_checkpoint(checkpoint_path=CKPT_PATH, batch_size=5, model=model,
                                               optimizer=opt, alpha=0.75)

    # Create metrics dict list, where results are logged
    metrics_dict_list = []
    metrics = smp.utils.metrics.IoU(threshold=0.5)
    cmap = 'gray'
    classes = ['cells', 'matrix', 'channels', 'cells-free zone']

    valid_preds = []
    for j, valid_sample in enumerate(test_loader):
        # valid_sample = next(iter(test_loader))
        test_model.eval()
        with torch.no_grad():
            prds = test_model.forward(valid_sample[0])

        valid_preds.append((valid_sample[1][0], prds[0]))

        metrics_dict = {}
        metrics_dict[j] = metrics(prds[0], valid_sample[1][0]).item()
        plt.figure(figsize=(7, 10))
        for i in range(4):
            class_iou = metrics(prds[0][i], valid_sample[1][0][i]).item()
            metrics_dict[f"{j}_{i}"] = class_iou

            plt.subplot(4, 2, i * 2 + 1)
            plt.title(f'Annotated {classes[i]}')
            plt.imshow(valid_sample[1][0][i], cmap=cmap)
            plt.axis('off')

            plt.subplot(4, 2, i * 2 + 2)
            plt.title(f'Predicted {classes[i]}')
            plt.imshow(prds[0][i], cmap=cmap)
            plt.axis('off')

        metrics_dict_list.append(metrics_dict)
        plt.suptitle(f"Total IoU score for image {j + 1} is {round(metrics_dict[j] * 100, 2)} %")

        output_dir = args.report_images_output_dir
        output_path = output_dir + f'/{args.split}_sample_{j+1}.png'
        plt.savefig(
            output_path,
            dpi=300, bbox_inches='tight')
        plt.show()
        
# Save results if necessary parameters are not None
if args.metrics_dict_list_save_path is not None:
    with open(args.metrics_dict_list_save_path, 'wb') as f:
        pkl.dump(metrics_dict_list, f)

if args.valid_preds_save_path is not None:
    with open(args.valid_preds_save_path, 'wb') as f:
        pkl.dump(valid_preds, f)
        