import json
from PIL import Image
from torch.utils.data import Dataset

from data import ImageTransforms


class SRDataset(Dataset):
    def __init__(self, crop_size, scaling_factor, lr_img_type, hr_img_type, 
                 augments={'rotation': False, 'hflip': False}, train_data_name=None):
        self.augments = augments
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.train_data_name = train_data_name

        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        with open(self.train_data_name, 'r') as j:
            self.images = json.load(j)

        # Select the correct set of transforms
        self.transform = ImageTransforms(crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type,
                                         augments=self.augments)

    def __getitem__(self, i):
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.images)
