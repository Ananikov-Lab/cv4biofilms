import numpy as np
import albumentations as albu
from torch.utils.data import Dataset, DataLoader


class SegmentedImagesMultiLabel(Dataset):
    def __init__(self, images, masks, classes, augmentation, crop_size=None):
        self.images = images
        self.masks = masks

        self.len = len(images)
        self.size = (images[0].shape[0], images[0].shape[1], 1)
        self.masks_size = (images[0].shape[0], images[0].shape[1], classes)

        self.augmentation = augmentation
        self.crop_size = crop_size

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.images[index].reshape(self.size).astype('float32') / 255
        mask = self.masks[index].reshape(self.masks_size).astype('float32')
        if self.crop_size is None:
            augmented = self.augmentation()(image=image, mask=mask)
        else:
            augmented = self.augmentation(self.crop_size)(image=image, mask=mask)

        return augmented['image'].transpose(2, 0, 1).astype('float32'), augmented['mask'].transpose(2, 0, 1).astype(
            'float32')


def get_loader_multilabel(dset_class, images, masks, classes, indexes, aug, batch_size, crop_size=None,
                          shuffle=True,
                          num_workers=1):
    """Creates corresponding dataloader
    """
    loader = DataLoader(
        dset_class(images[indexes], masks[indexes], classes, aug, crop_size=crop_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader
