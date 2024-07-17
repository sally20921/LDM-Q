import argparse, os, glob, yaml
import torch
import time
import numpy as np
import os
from torchvision.utils import save_image

from tqdm import trange

from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import torch.multiprocessing as mp
from ldm.models.diffusion.ddim import DDIMSampler

from scripts.sample_diffusion import custom_to_pil, custom_to_np

from ldm.models.diffusion.ddpm import LatentDiffusion

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step

@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    
    if not make_prog_row:
        return model.p_sample_loop(cond=None, shape=shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            cond=None, shape=shape, verbose=True
        )

@torch.no_grad()
def convsample_ddim(model, steps, shape, log_every_t=100, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, log_every_t=log_every_t, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, log_every_t=100, eta=1.0):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    
    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model=model, shape=shape, make_prog_row=False) # origin True LJW
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, log_every_t=log_every_t, 
                                                    shape=shape, eta=eta)
        t1 = time.time()

    if isinstance(model, LatentDiffusion):
        sample = model.decode_first_stage(sample)
    
    log["sample"] = sample
    # log["intermediates"] = intermediates
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def save_batch(batch, path, n_saved=0, np_path=None):
    if np_path is None:
        for x in batch:
            img = custom_to_pil(x)
            imgpath = os.path.join(path, f"{n_saved:06}.png")
            img.save(imgpath)
            n_saved += 1
    else:
        npbatch = custom_to_np(batch)
        shape_str = "x".join([str(x) for x in npbatch.shape])
        nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
        np.savez(nppath, npbatch)
        n_saved += npbatch.shape[0]
            
    return n_saved

def save_n_samples(model, n_samples, batch_size, n_saved, logdir, opt, target_step=None, log_every_t=100, key="sample"):
    if key == "intermediates":
        log_every_t = 1
    
    all_images = []
    for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
        logs = make_convolutional_sample(model, batch_size=batch_size, vanilla=opt.vanilla, log_every_t=log_every_t,
                                         custom_steps=opt.custom_steps, eta=opt.eta)
        if key == "intermediates":
            batch = logs[key]['x_inter'][target_step]
        else:
            batch = logs[key]
        
        n_saved = save_batch(batch, logdir, n_saved=n_saved)
        # all_images.extend([custom_to_np(batch)])
    
    left = n_samples % batch_size
    if left > 0:
        logs = make_convolutional_sample(model, batch_size=left, vanilla=opt.vanilla, log_every_t=log_every_t,
                                         custom_steps=opt.custom_steps, eta=opt.eta)
        if key == "intermediates":
            batch = logs[key]['x_inter'][target_step]
        else:
            batch = logs[key]
        n_saved = save_batch(batch, logdir, n_saved=n_saved)
        # all_images.extend([custom_to_np(batch)])

    return all_images, n_saved
    
def run(args):
    idx, model, gpu, opt = args
    
    torch.cuda.set_device(f"cuda:{gpu}")
    model.cuda()
    model.eval()
    
    n_samples = opt.n_samples
    batch_size = opt.batch_size

    if opt.vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {opt.custom_steps} sampling steps and eta={opt.eta}')
    
    n_saved = opt.n_start
    
    if model.cond_stage_model is None:
        print(f"[Device {gpu}] : Running unconditional sampling for {n_samples} samples")

        logdir = opt.logdir
        os.makedirs(logdir, exist_ok=True)
            
        all_images, _ = save_n_samples(model, n_samples, batch_size, n_saved, logdir, opt) 

    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')


def parse(parser):
    opt, unknown = parser.parse_known_args()

    opt.base = [opt.config]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    opt.config = OmegaConf.merge(*configs, cli)
    print(opt.config)
    
    print(75 * "=")
    print("logging to:")
    opt.logdir = os.path.join(opt.logdir, opt.name)

    os.makedirs(opt.logdir,   exist_ok=True)
    print(opt.logdir)
    print(75 * "=")
    
    return opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ckpt", type=str, nargs="?", help="load from logdir or checkpoint in logdir")
    parser.add_argument("--config", type=str, nargs="?", help="model config")
    
    parser.add_argument("-l", "--logdir", type=str, nargs="?", default="none", help="extra logdir")
    parser.add_argument("-n", "--n_samples", type=int, nargs="?", default=5, help="number of samples to draw")
    parser.add_argument("-v", "--vanilla", default=False, action='store_true', help="vanilla sampling (default option is DDIM sampling)?")
    parser.add_argument("-c", "--custom_steps", type=int, nargs="?", default=200, help="number of steps for ddim and fastdpm sampling")
    parser.add_argument("-e", "--eta", type=float, nargs="?", default=0.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
        
    parser.add_argument("-g", "--gpus", type=IndentationError, const=True, default=None, nargs="?", help="number of gpu to use")
    parser.add_argument("-d", "--dataset", type=str, const=True, default="lsun_churches256", nargs="?", help="dataset name")
    parser.add_argument("--name", type=str, const=True, default="_", nargs="?", help="dest folder name")
    parser.add_argument("-b", "--batch_size", type=int, const=True, default=32, nargs="?", help="batch size")
    
    parser.add_argument("--n_bit_w", type=int, nargs="?", default=8, const=True)
    parser.add_argument("--n_bit_a", type=int, nargs="?", default=8, const=True)
    
    parser.add_argument("--n_start", type=int, nargs="?", default=0, help="number of samples to start")
    parser.add_argument("--use_dynamic", action='store_true',help="Learning dynamic step size")
    
    opt = parse(parser)     
    
    tstart = time.time()
    
    model, _ = load_model(opt.config, opt.ckpt)
    
    model.model_ema.copy_to(model.model)
    model.use_ema = False
    
    # run
    run([0, model, opt.gpus, opt])

    print(f"sampling of {opt.n_samples} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")
