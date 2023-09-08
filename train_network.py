import pickle as pkl
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

from network_training.dataset_preparation import SegmentedImagesMultiLabel, get_loader_multilabel
from network_training.augmentations import get_training_augmentation, get_validation_augmentation


class SegModel(pl.LightningModule):

    def __init__(
            self,
            batch_size,
            model,
            alpha,
            optimizer,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = model
        self.optimizer = optimizer
        self.bce_loss = smp.utils.losses.BCELoss()
        self.dice_loss = smp.utils.losses.DiceLoss()

        self.metrics = smp.utils.metrics.IoU(threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.forward(x)
        alpha = self.alpha
        loss = alpha * self.bce_loss(prediction, y) + (1 - alpha) * self.dice_loss(prediction, y)
        metrics = self.metrics(prediction, y)
        self.log('train_loss', loss)
        self.log('train_metrics', metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.forward(x)
        alpha = self.alpha
        loss = alpha * self.bce_loss(prediction, y) + (1 - alpha) * self.dice_loss(prediction, y)
        metrics = self.metrics(prediction, y)
        self.log('valid_loss', loss)
        self.log('valid_metrics', metrics)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--path", type=str, help="Path to pickle, containing dataset")
    parser.add_argument("--arch_name", type=str, help="Model Architecture")
    parser.add_argument("--encoder_name", type=str, help="Model Encoder")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--optimizer_name", type=str, help="Optimizer name")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="special parameter in loss function")
    parser.add_argument("--crop_size", type=int, default=896, help="Crop size in augmentations")
    parser.add_argument("--savelogdir", type=str, help="Path, where model is saved")
    parser.add_argument("--train_size", type=int, help="Train size")
    parser.add_argument("--elastic_transform_size", type=int, help="Number of elastic transforms in dataloader")
    parser.add_argument("--substrate_filter", type=int, default=0,
                        help="If 1, script is used to train substrate filter model")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of segmentation classes")

    args = parser.parse_args()

    # Set model hyperparameters
    architectures = {
        'UNet': smp.Unet,
        'FPN': smp.FPN,
        'PSPNet': smp.PSPNet
    }

    arch = architectures[args.arch_name]
    classes = args.n_classes

    model = arch(args.encoder_name, in_channels=1, classes=classes, activation='sigmoid')

    optimizers = {
        'Adam': torch.optim.Adam(params=model.parameters(), lr=args.lr),
        'SGD': torch.optim.SGD(params=model.parameters(), lr=args.lr),
        'RMSprop': torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    }
    optimizer = optimizers[args.optimizer_name]

    if not args.substrate_filter:
        images = []
        masks = []
        with open(args.path, 'rb') as f:
            for item in pkl.load(f)[:10]:
                if not item:
                    continue

                if any([n in item['name'] for n in ['103', '020', '018']]):
                    continue

                images.append(item['image'][:-140])
                if args.substrate_filter:
                    masks.append(item['mask'][:-140, :, 3])
                else:
                    masks.append(item['mask'][:-140])

        indexes = np.arange(len(images))
        
        # If use elastic transforms, open pickle file with elastic transformed images
        if args.elastic_transform_size != 0:
            with open('/home/kskozlov/files_for_research/bacteria_img_analysis/elastic_augs.pkl', 'rb') as f:
                elastic_transform_data = pkl.load(f)

        train, test = train_test_split(indexes, test_size=0.1, random_state=42)
        train, val = train_test_split(train, test_size=0.1, random_state=42)

        train_images = np.array([images[item] for item in train])[:args.train_size]
        train_masks = np.array([masks[item] for item in train])[:args.train_size]

        if args.elastic_transform_size != 0:
            elastic_images, elastic_masks = elastic_transform_data
            elastic_images = np.array(elastic_images[:args.elastic_transform_size])
            elastic_masks = np.array(elastic_masks[:args.elastic_transform_size])

            train_images = np.concatenate((train_images, elastic_images), axis=0)
            train_masks = np.concatenate((train_masks, elastic_masks), axis=0)

        test_images = np.array([images[item] for item in val])
        test_masks = np.array([masks[item] for item in val])
        
    else:
        with open(args.path, 'rb') as f:
            substrate_dataset = pkl.load(f)
            train_images = substrate_dataset['x_train']
            test_images = substrate_dataset['x_test']

            train_masks = substrate_dataset['y_train']
            test_masks = substrate_dataset['y_test']

    # Place datasets into dataloaders
    train_dataloader = get_loader_multilabel(SegmentedImagesMultiLabel,
                                             train_images, train_masks, classes,
                                             np.arange(len(train_images)),
                                             get_training_augmentation, args.batch_size, crop_size=args.crop_size)

    val_dataloader = get_loader_multilabel(SegmentedImagesMultiLabel, test_images, test_masks, classes,
                                           np.arange(len(test_images)), get_validation_augmentation, shuffle=False,
                                           batch_size=1)

    # Train epochs and specify tensorboard logging directory
    seg_model = SegModel(batch_size=args.batch_size,
                         alpha=args.alpha,
                         model=model,
                         optimizer=optimizer)

    tb_logger = pl.loggers.TensorBoardLogger(f"logs/{args.savelogdir}/{args.elastic_transform_size}")
    trainer = pl.Trainer(gpus=1, max_epochs=300, logger=tb_logger)
    trainer.fit(seg_model, train_dataloader, val_dataloader)
