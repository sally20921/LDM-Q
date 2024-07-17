import os
import argparse
import glob
from PIL import Image
from omegaconf import OmegaConf

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torch_fidelity
#from scripts.sample_diffusion import instantiate_from_config
from sample_diffusion import instantiate_from_config

class EvalDataset(Dataset):
    def __init__(self, sample_path):
        self.image_paths = glob.glob(sample_path + "/*.png")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = transforms.functional.pil_to_tensor(img)
        return img


def get_dataset(opt):
    if opt.dataset == 'cifar-10' or opt.dataset == 'CIFAR-10':
        return 'cifar10-train'
    
    elif opt.dataset == 'lsun_churches256' or opt.dataset == 'LSUN_churches256' or opt.dataset == 'lsun_churches' or opt.dataset == 'LSUN-churches':
        opt.dataset = 'lsun_churches256'
        config_path = "models/ldm/lsun_churches256/config.yaml"
        
        config = OmegaConf.load(config_path)

        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()

        return data.datasets['train']
    
    else:
        raise NotImplementedError(f"Not Supported dataset {opt.dataset}")


def validate(opt):
    dataset1 = EvalDataset(opt.sample_path)
    dataset2 = get_dataset(opt)
    print(f"Dataset Length : {len(dataset1)}, {len(dataset2)}")

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dataset1, 
        input2=dataset2,
        batch_size=256,
        cache_root = 'fidelity_cache',
        cache = True,
        # input1_cache_name = opt.dataset + "_" + opt.name,
        input2_cache_name = opt.dataset,
        datasets_root = '/SSD',
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=False, 
        samples_shuffle=False,
        verbose=True,
    )
    print("=" * 75)
    print(metrics_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--logdir", type=str, const=True, default="/SSD/stable_diffusion/QAT/samples", nargs="?", help="PTQ model")
    parser.add_argument("-d", "--dataset", type=str, const=True, default="lsun_churches256", nargs="?", help="dataset name")
    parser.add_argument("-g", "--gpu", type=int, const=True, default=0, nargs="?", help="gpu number")
    parser.add_argument("-n", "--name", type=str, const=True, default="FP32", nargs="?", help="PTQ model")
    
    opt = parser.parse_args()        

    opt.sample_path = os.path.join(opt.logdir, opt.name)
    print("Sample Path : ", opt.sample_path)
    
    torch.cuda.set_device(f'cuda:{opt.gpu}')

    print(f"[Device {opt.gpu}] Valiate dataset {opt.dataset}-{opt.name}")
    validate(opt)
