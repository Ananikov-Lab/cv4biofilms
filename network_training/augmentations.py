import albumentations as albu
import cv2


def get_training_augmentation(crop_size=256):
    train_transform = [
        albu.HorizontalFlip(p=0.5),

        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=(-0.5, 1), rotate_limit=45,
                              shift_limit=0.6, p=1, border_mode=cv2.BORDER_REFLECT),

        albu.RandomCrop(crop_size, crop_size),

        albu.GridDistortion(),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
                albu.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=1),
            ],
            p=0.9,
        )

    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.CenterCrop(1760, 2560)
    ]
    return albu.Compose(test_transform)
