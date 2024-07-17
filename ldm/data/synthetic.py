import torch
import torchvision
from torchvision import transforms
from einops import rearrange
import glob
import os
import PIL
from PIL import Image
import numpy as np

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root, size=None, interpolation="bicubic", flip_p=0.5):
        self.image_path = glob.glob(os.path.join(root, "*.png"))

        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        image = np.array(image).astype(np.uint8)

        image = Image.fromarray(image)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return {"image": image}
    
    def __len__(self):
        return len(self.image_path)


class SyntheticDataset2(torch.utils.data.Dataset):
    def __init__(self, root, size=None, interpolation="bicubic", flip_p=0.5):
        self.image_path = glob.glob(os.path.join(root, "*.png"))

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        assert image.shape == (3, 32, 32)
        image = rearrange(image, 'c h w -> h w c')
        return {"image": image}
    
    def __len__(self):
        return len(self.image_path)
        
if __name__ == '__main__':
    SyntheticDataset(root='/SSD/stable_diffusion/samples/checkpoints/35M_390K/')