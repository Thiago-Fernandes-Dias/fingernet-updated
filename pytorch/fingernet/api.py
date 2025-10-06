import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

from .model import get_fingernet, DEFAULT_WEIGHTS_PATH

torch.set_float32_matmul_precision('medium')

class FingerprintDataset(Dataset):
    """Dataset for loading fingerprint images."""
    
    def __init__(self, image_paths: list[str], target_size: tuple[int, int]):
        self.image_paths = image_paths
        self.target_size = target_size  # (H, W)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('L')
        img_np = np.array(img_pil, dtype=np.float32) / 255.0

        # Padding to target_size (white background = 1.0)
        h, w = img_np.shape
        th, tw = self.target_size
        pad_h = th - h
        pad_w = tw - w
        
        if pad_h < 0 or pad_w < 0:
            # Resize to fit target_size, maintaining aspect ratio
            scale = min(th / h, tw / w)
            nh, nw = int(h * scale), int(w * scale)
            img_pil = img_pil.resize((nw, nh), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            h, w = img_np.shape
            pad_h = th - h
            pad_w = tw - w
            
        # Apply padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img_np = np.pad(
            img_np, 
            ((pad_top, pad_bottom), (pad_left, pad_right)), 
            mode='constant', 
            constant_values=1.0
        )
        img_np = np.ascontiguousarray(img_np)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        return img_tensor, img_path

def find_image_paths(input_path: str, recursive: bool = False) -> list[str]:
    """
    Find all image paths from input.
    
    Args:
        input_path: Path to file, directory, or text list
        recursive: Whether to search recursively in directories
        
    Returns:
        List of image paths
    """
    image_paths = []
    
    if os.path.isfile(input_path):
        # Check if it's a text file (list of paths)
        _, ext = os.path.splitext(input_path)
        if ext.lower() in ['.txt', '.list']:
            with open(input_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path:
                        image_paths.append(path)
        else:
            image_paths.append(input_path)
            
    elif os.path.isdir(input_path):
        extensions = ['png', 'bmp', 'jpg', 'jpeg']
        for ext in extensions:
            pattern = f"{input_path}/**/*.{ext}" if recursive else f"{input_path}/*.{ext}"
            image_paths.extend(glob.glob(pattern, recursive=recursive))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
        
    if not image_paths:
        raise ValueError(f"No images found in: {input_path}")
        
    return sorted(image_paths)


def get_image_dimensions(image_paths: list[str]) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get minimum and maximum dimensions from image list.
    
    Returns:
        (min_shape, max_shape) where each is (height, width)
    """
    min_h, min_w = float('inf'), float('inf')
    max_h, max_w = 0, 0
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                min_h = min(min_h, h)
                min_w = min(min_w, w)
                max_h = max(max_h, h)
                max_w = max(max_w, w)
        except Exception as e:
            print(f"Warning: Error opening {img_path}: {e}")
            continue
            
    return (min_h, min_w), (max_h, max_w)

def save_results(result_item: dict, output_path: str, mnt_degrees: bool = False):
    """
    Save inference results to disk in organized structure.
    
    Args:
        result_item: Dictionary with keys 'input_path', 'minutiae', 'enhanced_image', etc.
        output_path: Base output directory
        mnt_degrees: If True, save minutiae angles in degrees instead of radians
    """
    original_filename = os.path.basename(result_item['input_path'])
    base_name = os.path.splitext(original_filename)[0]

    # Save minutiae (.txt)
    minutiae = result_item['minutiae'].copy()
    if mnt_degrees:
        minutiae[:, 2] = np.round(np.rad2deg(minutiae[:, 2]), 2)
    
    minutiae_path = os.path.join(output_path, 'minutiae', f"{base_name}.txt")
    np.savetxt(
        minutiae_path, 
        minutiae, 
        fmt=['%.0f', '%.0f', '%.6f', '%.6f'], 
        header='x, y, angle, score', 
        delimiter=','
    )

    # Save enhanced image (.png)
    enhanced_path = os.path.join(output_path, 'enhanced', original_filename)
    Image.fromarray(result_item['enhanced_image']).save(enhanced_path)

    # Save mask (.png)
    mask_path = os.path.join(output_path, 'mask', original_filename)
    Image.fromarray(result_item['segmentation_mask']).save(mask_path)

    # Save orientation field (encoded as PNG)
    ori_cpu = result_item['orientation_field']
    orientation_path = os.path.join(output_path, 'ori', original_filename)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)


def create_output_directories(output_path: str):
    """Create output directory structure."""
    os.makedirs(os.path.join(output_path, 'minutiae'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'ori'), exist_ok=True)

def save_results(result_item: dict, output_path: str, mnt_degrees: bool = False):
    """
    Save inference results to disk in organized structure.
    
    Args:
        result_item: Dictionary with keys 'input_path', 'minutiae', 'enhanced_image', etc.
        output_path: Base output directory
        mnt_degrees: If True, save minutiae angles in degrees instead of radians
    """
    original_filename = os.path.basename(result_item['input_path'])
    base_name = os.path.splitext(original_filename)[0]

    # Save minutiae (.txt)
    minutiae = result_item['minutiae'].copy()
    if mnt_degrees:
        minutiae[:, 2] = np.round(np.rad2deg(minutiae[:, 2]), 2)
    
    minutiae_path = os.path.join(output_path, 'minutiae', f"{base_name}.txt")
    np.savetxt(
        minutiae_path, 
        minutiae, 
        fmt=['%.0f', '%.0f', '%.6f', '%.6f'], 
        header='x, y, angle, score', 
        delimiter=','
    )

    # Save enhanced image (.png)
    enhanced_path = os.path.join(output_path, 'enhanced', original_filename)
    Image.fromarray(result_item['enhanced_image']).save(enhanced_path)

    # Save mask (.png)
    mask_path = os.path.join(output_path, 'mask', original_filename)
    Image.fromarray(result_item['segmentation_mask']).save(mask_path)

    # Save orientation field (encoded as PNG)
    ori_cpu = result_item['orientation_field']
    orientation_path = os.path.join(output_path, 'ori', original_filename)
    angles_deg_shifted = np.round(np.rad2deg(ori_cpu) + 90).astype(np.uint8)
    Image.fromarray(angles_deg_shifted).save(orientation_path)


def create_output_directories(output_path: str):
    """Create output directory structure."""
    os.makedirs(os.path.join(output_path, 'minutiae'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'ori'), exist_ok=True)

def setup_ddp(rank: int, world_size: int, timeout_minutes: int = 30):
    """
    Initialize distributed process group.
    
    Args:
        rank: Unique identifier for this process
        world_size: Total number of processes
        timeout_minutes: Timeout for DDP operations
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Set device for this process BEFORE init_process_group
    torch.cuda.set_device(rank)
    
    # Initialize process group with proper timeout and device_id
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device(f'cuda:{rank}')
    )

def cleanup_ddp():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
    

def inference_worker_ddp(
    rank: int,
    world_size: int,
    image_paths: list[str],
    target_size: tuple[int, int],
    output_path: str,
    weights_path: str,
    batch_size: int,
    num_workers: int,
    mnt_degrees: bool,
    compile_model: bool
):
    """
    Worker function for distributed inference on a single GPU.
    
    Args:
        rank: GPU rank/ID for this worker
        world_size: Total number of GPUs
        image_paths: List of all image paths
        target_size: Target size for images (H, W)
        output_path: Output directory
        weights_path: Path to model weights
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        mnt_degrees: Save minutiae angles in degrees
        compile_model: Whether to compile model with torch.compile
    """
    try:
        # Setup DDP
        setup_ddp(rank, world_size)
        
        # Only rank 0 prints and creates directories
        is_main = (rank == 0)
        
        if is_main:
            # print(f"\n{'='*60}")
            print(f"Starting Distributed Inference")
            # print(f"{'='*60}")
            print(f"World Size: {world_size} GPUs")
            print(f"Batch Size per GPU: {batch_size}")
            print(f"Total Images: {len(image_paths)}")
            print(f"Output Path: {output_path}")
            # print(f"{'='*60}\n")
            
            # Create output directories
            create_output_directories(output_path)
        
        # Synchronize: all processes wait for rank 0 to create directories
        dist.barrier()
        
        # Load model
        if is_main:
            print(f"[Rank {rank}] Loading model...")
        
        model = get_fingernet(weights_path=weights_path, device=f'cuda:{rank}', log=False)
        model.eval()
        
        # Optionally compile model
        if compile_model:
            if is_main:
                print(f"[Rank {rank}] Compiling model with torch.compile...")
            model = torch.compile(model)
        
        # Wrap with DDP - not strictly necessary for inference-only but helps with synchronization
        # model = DDP(model, device_ids=[rank])
        
        # Setup dataset and distributed sampler
        dataset = FingerprintDataset(image_paths, target_size)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        if is_main:
            print(f"[Rank {rank}] Starting inference...")
        
        # Inference loop
        all_results = []
        
        with torch.no_grad():
            # Use tqdm only on main process
            iterator = tqdm(dataloader, desc=f"GPU {rank}") if is_main else dataloader
            
            for batch_tensors, batch_paths in iterator:
                # Move to device
                batch_tensors = batch_tensors.to(f'cuda:{rank}')
                
                # Forward pass
                results = model(batch_tensors)
                
                # Process each image in batch
                for i in range(len(batch_paths)):
                    result_item = {
                        'input_path': batch_paths[i],
                        'minutiae': results['minutiae'][i].cpu().numpy(),
                        'enhanced_image': results['enhanced_image'][i].cpu().numpy(),
                        'segmentation_mask': results['segmentation_mask'][i].cpu().numpy(),
                        'orientation_field': results['orientation_field'][i].cpu().numpy(),
                    }
                    all_results.append(result_item)
        
        # Synchronize: all processes finish inference before saving
        dist.barrier()
        
        if is_main:
            print(f"\n[Rank {rank}] Inference complete. Saving results...")
        
        # Each rank saves its own results
        for result_item in tqdm(all_results, desc=f"Saving (GPU {rank})", disable=not is_main):
            save_results(result_item, output_path, mnt_degrees)
        
        # Final synchronization
        print(f"[Rank {rank}] Finished saving results.")
        dist.barrier()
        
        # ALL processes must participate in reduce (collective operation)
        total_processed = torch.tensor(len(all_results), device=f'cuda:{rank}')
        dist.reduce(total_processed, dst=0, op=dist.ReduceOp.SUM)
        
        # Only rank 0 prints the results
        if is_main:
            total = total_processed.item()
            print(f"\n{'='*60}")
            print(f"✓ Inference Complete!")
            print(f"  Total images processed: {total}")
            print(f"  Results saved to: {output_path}")
            print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_ddp()

def inference_single_gpu(
    device_id: int,
    image_paths: list[str],
    target_size: tuple[int, int],
    output_path: str,
    weights_path: str,
    batch_size: int,
    num_workers: int,
    mnt_degrees: bool,
    compile_model: bool
):
    """
    Inference on a single GPU (no DDP).
    
    Args:
        device_id: GPU ID to use
        image_paths: List of image paths
        target_size: Target size (H, W)
        output_path: Output directory
        weights_path: Path to model weights
        batch_size: Batch size
        num_workers: Number of data loading workers
        mnt_degrees: Save angles in degrees
        compile_model: Whether to compile model
    """
    print(f"\n{'='*60}")
    print(f"Starting Single GPU Inference")
    print(f"{'='*60}")
    print(f"Device: cuda:{device_id}")
    print(f"Batch Size: {batch_size}")
    print(f"Total Images: {len(image_paths)}")
    print(f"Output Path: {output_path}")
    print(f"{'='*60}\n")
    
    # Create output directories
    create_output_directories(output_path)
    
    # Load model
    print("Loading model...")
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    model = get_fingernet(weights_path=weights_path, device=device, log=False)
    model.eval()
    
    # Optionally compile
    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Setup dataset and dataloader
    dataset = FingerprintDataset(image_paths, target_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=(num_workers > 0)
    )
    
    print("Starting inference...")
    
    # Inference loop
    all_results = []
    
    with torch.no_grad():
        for batch_tensors, batch_paths in tqdm(dataloader, desc="Processing"):
            batch_tensors = batch_tensors.to(device)
            results = model(batch_tensors)
            
            for i in range(len(batch_paths)):
                result_item = {
                    'input_path': batch_paths[i],
                    'minutiae': results['minutiae'][i].cpu().numpy(),
                    'enhanced_image': results['enhanced_image'][i].cpu().numpy(),
                    'segmentation_mask': results['segmentation_mask'][i].cpu().numpy(),
                    'orientation_field': results['orientation_field'][i].cpu().numpy(),
                }
                all_results.append(result_item)
    
    print("\nSaving results...")
    for result_item in tqdm(all_results, desc="Saving"):
        save_results(result_item, output_path, mnt_degrees)
    
    print(f"\n{'='*60}")
    print(f"✓ Inference Complete!")
    print(f"  Total images processed: {len(all_results)}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}\n")

def run_inference(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    recursive: bool = False,
    mnt_degrees: bool = False,
    compile_model: bool = False
):
    """
    Run FingerNet inference on images.
    
    Args:
        input_path: Path to image, directory, or text file with image paths
        output_path: Directory to save results
        weights_path: Path to model weights (.pth file)
        gpus: GPU configuration:
            - None or 0: Use CPU
            - int (e.g., 1): Use single GPU with ID 0
            - int (e.g., 2): Use 2 GPUs with DDP (IDs 0,1)
            - list[int] (e.g., [2,3]): Use specific GPUs with DDP
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers per GPU
        recursive: Search for images recursively
        mnt_degrees: Save minutiae angles in degrees instead of radians
        compile_model: Use torch.compile for faster inference
        
    Example:
        >>> run_inference('images/', 'output/', gpus=2, batch_size=8)
    """
    # Find all images
    # print("Discovering images...")
    image_paths = find_image_paths(input_path, recursive)
    
    # Get image dimensions
    min_shape, max_shape = get_image_dimensions(image_paths)
    # print(f"Image dimensions: min={min_shape}, max={max_shape}")
    # print(f"Using target size: {max_shape}")
    
    # Parse GPU configuration
    if gpus is None or gpus == 0:
        # CPU mode
        print("Running on CPU...")
        device = 'cpu'
        inference_single_gpu(
            device_id=-1,
            image_paths=image_paths,
            target_size=max_shape,
            output_path=output_path,
            weights_path=weights_path,
            batch_size=batch_size,
            num_workers=num_workers,
            mnt_degrees=mnt_degrees,
            compile_model=compile_model
        )
        
    elif isinstance(gpus, int):
        if gpus == 1:
            # Single GPU
            inference_single_gpu(
                device_id=0,
                image_paths=image_paths,
                target_size=max_shape,
                output_path=output_path,
                weights_path=weights_path,
                batch_size=batch_size,
                num_workers=num_workers,
                mnt_degrees=mnt_degrees,
                compile_model=compile_model
            )
        else:
            # Multiple GPUs with DDP (use GPUs 0 to gpus-1)
            world_size = gpus
            mp.spawn(
                inference_worker_ddp,
                args=(
                    world_size,
                    image_paths,
                    max_shape,
                    output_path,
                    weights_path,
                    batch_size,
                    num_workers,
                    mnt_degrees,
                    compile_model
                ),
                nprocs=world_size,
                join=True
            )
            
    elif isinstance(gpus, list):
        # Specific GPU IDs with DDP
        world_size = len(gpus)
        # Set visible devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
        mp.spawn(
            inference_worker_ddp,
            args=(
                world_size,
                image_paths,
                max_shape,
                output_path,
                weights_path,
                batch_size,
                num_workers,
                mnt_degrees,
                compile_model
            ),
            nprocs=world_size,
            join=True
        )
    else:
        raise ValueError(f"Invalid gpus parameter: {gpus}")

def run_enhancement(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    recursive: bool = False
):
    """
    Run only image enhancement (faster than full inference).
    
    Note: This is a placeholder. Full implementation would require
    modifying the worker functions to call model.enhance() instead.
    """
    print("Enhancement-only mode not yet implemented.")
    print("Use run_inference() for now.")
    raise NotImplementedError("Enhancement-only mode coming soon")


def run_segmentation(
    input_path: str,
    output_path: str,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    gpus: int | list[int] | None = None,
    batch_size: int = 4,
    recursive: bool = False
):
    """
    Run only segmentation (faster than full inference).
    
    Note: This is a placeholder. Full implementation would require
    modifying the worker functions to call model.segment() instead.
    """
    print("Segmentation-only mode not yet implemented.")
    print("Use run_inference() for now.")
    raise NotImplementedError("Segmentation-only mode coming soon")