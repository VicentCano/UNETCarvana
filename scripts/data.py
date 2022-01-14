import os
from PIL import Image
from torch.utils.data import Dataset
import numpy

class Carvana(Dataset):
    def __init__(self, images_folder, masks_folder, transform = None):
        self.images_folder = images_folder
        self.mask_folder = masks_folder
        self.transform = transform
        self.images = os.listdir(images_folder)

    def __len__(self):
        return  len(self.images)
    def __getitem__(self, i):
        img_path = os.path.join(self.images_folder, self.images[i])
        mask_path = os.path.join(self.mask_folder, self.images[i].replace(".jpg", "_mask.gif"))
        image = numpy.array(Image.open(img_path).convert("RGB"))
        mask = numpy.array(Image.open(mask_path).convert("L"), dtype = numpy.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return  image, mask
