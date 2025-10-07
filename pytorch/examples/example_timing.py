import os.path as osp
import os

import torch
import fingernet as fnet
import time
from PIL import Image
import re
from tqdm import tqdm
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# sd258_000_11-00_latent_bad.png
LATENT_REGEXP = r'.+-00'
# sd258_000_11-01_template_bad.png
REFERENCE_REGEXP = r'.+-01'

MODEL = 'wrapper' # options are: 'core', 'wrapper'
TOTAL = None

def list_images(path, regexp=None, recursive=False, total=TOTAL):
    # Select all files in path
    if osp.isfile(path):
        return [path]
    elif osp.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            if total is not None and len(files) >= total:
                break
            
            for filename in filenames:
                if regexp is None or re.match(regexp, filename):
                    files.append(osp.join(root, filename))
                    if total is not None and len(files) >= total:
                        break
            
            if not recursive:
                break
        return files

def load_images(image_paths, to_tensor=False):
    images = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('L')
        if to_tensor:
            image = transforms.ToTensor()(image)
        images.append(image)
    return images

def cpu_inference(cpus=1, **kwargs):
    """Run inference on CPU processing one image at a time.

    Pass `cpus` to control the number of threads (use 1 for single-core runs).
    """
    return _run_inference(device='cpu', batch_mode=False, cpus=cpus, **kwargs)

def cpu_batch_inference(cpus=1, **kwargs):
    """Run inference on CPU using batches provided by a DataLoader.

    The dataloader will handle the number of batches automatically. This
    function uses a lazy ImageDataset to avoid loading all images into memory
    at once. Pass `cpus=1` to force single-core execution.
    """
    return _run_inference(device='cpu', batch_mode=True, cpus=cpus, **kwargs)

def single_gpu_inference(device_id=0, **kwargs):
    """Run inference on a single GPU processing one image at a time.

    Falls back to CPU if CUDA is unavailable.
    """
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('CUDA not available, falling back to CPU for single_gpu_inference')
    return _run_inference(device=device, batch_mode=False, **kwargs)

def single_gpu_batch_inference(device_id=0, **kwargs):
    """Run inference on a single GPU using batched DataLoader.

    Falls back to CPU if CUDA is unavailable. DataLoader automatically
    determines number of batches from dataset size and batch_size.
    """
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('CUDA not available, falling back to CPU for single_gpu_batch_inference')
    return _run_inference(device=device, batch_mode=True, **kwargs)


class ImageDataset(Dataset):
    """Lazy-loading image dataset that returns a single-channel tensor."""
    def __init__(self, paths, to_tensor=True):
        self.paths = list(paths)
        self.to_tensor = to_tensor
        self.transform = transforms.ToTensor() if to_tensor else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img


def _make_dataloader(path, regexp=None, batch_size=8, num_workers=0, shuffle=False, recursive=False):
    """Helper to create a DataLoader from a path.

    If `path` is a file it will create a dataset with a single element.
    """
    image_paths = list_images(path, regexp=regexp, recursive=recursive)
    dataset = ImageDataset(image_paths, to_tensor=True)
    pin_memory = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _run_inference(device='cpu', batch_mode=True, batch_size=8, cpus=None, num_workers=0, compile=True):
    """Internal runner used by other functions.

    - device: 'cpu' or 'cuda'
    - batch_mode: if True use DataLoader batches, otherwise iterate single images
    - batch_size/num_workers: forwarded to DataLoader when batch_mode is True
    - compile: whether to torch.compile the model when available
    """
    # choose dataset
    latent_root = '/storage/jcontreras/data/datasets/SD258/orig'

    if batch_mode:
        loader = _make_dataloader(latent_root, regexp=LATENT_REGEXP, batch_size=batch_size, num_workers=num_workers, recursive=True)
        total_items = len(loader.dataset)
        print(f'Created dataloader with {total_items} images, batch_size={batch_size}, num_workers={num_workers}')
    else:
        # single image iterator using batch_size=1 DataLoader for consistent behavior
        loader = _make_dataloader(latent_root, regexp=LATENT_REGEXP, batch_size=1, num_workers=num_workers, recursive=True)
        total_items = len(loader.dataset)
        print(f'Created single-image dataloader with {total_items} images')

    # instantiate model on desired device
    if MODEL == 'wrapper':
        model = fnet.get_fingernet(device=device, log=True)
    else:
        model = fnet.get_fingernet_core(device=device, log=True)

    if compile and device == 'cpu':
        try:
            # If the model is on cpu, compile it for better performance
            model = torch.compile(model)
            print('Compiled model with torch.compile()')
        except Exception as e:
            print('torch.compile failed:', e)

    model.eval()

    # set CPU thread affinity when running on CPU
    if device == 'cpu':
        # if cpus is provided and positive, use it; otherwise leave PyTorch defaults
        if cpus is not None and cpus > 0:
            os.environ['OMP_NUM_THREADS'] = str(cpus)
            os.environ['MKL_NUM_THREADS'] = str(cpus)
            torch.set_num_threads(cpus)
            print(f'Running on CPU with {cpus} thread(s)')
        else:
            # reflect current PyTorch thread setting into env vars
            n = torch.get_num_threads()
            os.environ['OMP_NUM_THREADS'] = str(n)
            os.environ['MKL_NUM_THREADS'] = str(n)

    # run
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            if device.startswith('cuda'):
                # non_blocking means async copy if pinned memory
                batch = batch.to('cuda', non_blocking=True) 
            _ = model.time(batch)

    return True

def multi_gpu_inference():
    pass

def multi_gpu_batch_inference():
    pass

if __name__ == "__main__":
    #cpu_inference(cpus=1, compile=False)
    # single_gpu_inference()
    single_gpu_batch_inference(batch_size=16)