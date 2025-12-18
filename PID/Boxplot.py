import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib  # handle NIfTI files
from scipy.ndimage import label  # connected-component labeling

# =========================
# GPU Kernels (numba.cuda)
# =========================

from numba import cuda, float32, uint8
import math
import numpy as np

@cuda.jit(fastmath=True)
def depth_reduce_kernel(masks, original_center, area_center, depths):
    # One block processes one mask; 256 threads per block perform parallel reduction
    bid = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    n_masks = masks.shape[0]
    n_voxels = masks.shape[1]
    if bid >= n_masks:
        return

    sm = cuda.shared.array(shape=512, dtype=float32)  # 256*2
    sm_area = sm
    sm_dot  = sm[256:]

    partial_area = 0.0
    partial_dot  = 0.0

    for j in range(tid, n_voxels, bdim):
        v = float32(masks[bid, j])       
        c = original_center[j]              # f32
        partial_area += v
        partial_dot  += v * c

    sm_area[tid] = partial_area
    sm_dot[tid]  = partial_dot
    cuda.syncthreads()

    offset = bdim // 2
    while offset > 0:
        if tid < offset:
            sm_area[tid] += sm_area[tid + offset]
            sm_dot[tid]  += sm_dot[tid + offset]
        cuda.syncthreads()
        offset >>= 1  # halve each iteration until 1 to reduce across all threads


    if tid == 0:
        area_mask = sm_area[0]
        dot_c     = sm_dot[0]
        if area_mask > 0.0 and area_center > 0.0:
            s1 = dot_c / area_mask
            s2 = dot_c / area_center
            depths[bid] = s1 if s1 < s2 else s2
        else:
            depths[bid] = 0.0


def probabilistic_inclusion_depth_cpu(masks_data):
    """
    Compute Probabilistic Inclusion Depth (PID) on CPU.
    
    Use the identity to simplify computation:
    1 - sum(M*(1-C))/sum(M) === sum(M*C)/sum(M)
    """
    num_samples = masks_data.shape[0]
    # 1) Flatten data
    masks_flattened = masks_data.reshape(num_samples, -1)
    # 2) Compute center
    original_center = np.mean(masks_flattened, axis=0, dtype=np.float32)
    # 3) Compute denominators (areas)
    area_masks = np.sum(masks_flattened, axis=1, dtype=np.float32)
    area_center = np.sum(original_center, dtype=np.float32)
    # 4) Compute numerators (shared dot products)
    dot_product = np.sum(masks_flattened * original_center, axis=1, dtype=np.float32)
    # 5) Compute s1 and s2
    # s1 = (M·C) / |M|
    s1 = np.divide(dot_product, area_masks, out=np.zeros_like(dot_product), where=area_masks!=0)
    # s2 = (M·C) / |C|
    if area_center > 0:
        s2 = dot_product / area_center
    else:
        s2 = np.zeros_like(dot_product)
        
    # 6) Depth is the minimum
    depths = np.minimum(s1, s2)
    
    return depths.astype(np.float32, copy=False)
def probabilistic_inclusion_depth(masks_data, use_gpu=True, batch_size=128):
    # ===== CPU fallback =====
    if (not use_gpu) or (not cuda.is_available()):
        print("CUDA not enabled or no available GPU detected, automatically falling back to CPU computation.")
        return probabilistic_inclusion_depth_cpu(masks_data)

    # ===== GPU path (double buffering + two streams) =====
    print("\n=== GPU Acceleration (Double Buffering + Transfer/Compute Overlap) ===")
    # Total timer (includes pre-warm)
    t_total0 = time.time()

    # Preprocessing: flatten + compute center
    t0 = time.time()
    num_samples = masks_data.shape[0]
    masks_flat = masks_data.reshape(num_samples, -1)              # (N, V) float32 assumed
    original_center = masks_flat.mean(axis=0, dtype=np.float32)   # (V,)
    area_center = np.float32(original_center.sum(dtype=np.float32))
    V = masks_flat.shape[1]
    preprocessing_time = time.time() - t0

    # Two streams
    t_setup0 = time.time()
    stream_h2d = cuda.stream()
    stream_k   = cuda.stream()
    # Reusable H2D events (double buffering)
    ev_h2d = [cuda.event(timing=False), cuda.event(timing=False)]
    setup_streams_time = time.time() - t_setup0

    # Upload center once
    t0 = time.time()
    d_center = cuda.to_device(original_center, stream=stream_h2d)
    stream_h2d.synchronize()
    h2d_time = time.time() - t0  # for reference only

    # Pre-warm: trigger JIT compilation and init streams/events (excluded from main stats)
    t_pw0 = time.time()
    d_masks_pw = cuda.to_device(np.zeros((1, V), dtype=np.float32), stream=stream_h2d)
    d_out_pw = cuda.device_array(1, dtype=np.float32)
    ev_h2d[0].record(stream_h2d)
    ev_h2d[0].wait(stream_k)
    depth_reduce_kernel[1, 256, stream_k](d_masks_pw[:1, :], d_center, area_center, d_out_pw[:1])
    d_out_pw[:1].copy_to_host(cuda.pinned_array(1, dtype=np.float32), stream=stream_k)
    stream_k.synchronize()
    stream_h2d.synchronize()
    prewarm_time = time.time() - t_pw0

    # Result container
    all_depths = np.empty(num_samples, dtype=np.float32)

    # Config
    B = int(batch_size)
    threadsperblock = 256
    # Two device buffers + two pinned host buffers (input/output)
    t_alloc0 = time.time()
    d_masks = [cuda.device_array((B, V), dtype=np.float32),
               cuda.device_array((B, V), dtype=np.float32)]
    d_out   = [cuda.device_array(B, dtype=np.float32),
               cuda.device_array(B, dtype=np.float32)]
    h_in  = [cuda.pinned_array((B, V), dtype=np.float32),
             cuda.pinned_array((B, V), dtype=np.float32)]
    h_out = [cuda.pinned_array(B, dtype=np.float32),
             cuda.pinned_array(B, dtype=np.float32)]
    alloc_buffers_time = time.time() - t_alloc0

    # Reusable D2H completion events (double buffering)
    done_evt   = [None, None]
    done_slice = [None, None]   # (start_idx, length) record result range

    # Reset sub-timers after pre-warm (keep total timer including pre-warm)
    gpu_compute_time = 0.0
    gpu_h2d_time = 0.0
    gpu_d2h_time = 0.0
    h2h_copy_time = 0.0  # Host->Host copy time: all_depths <- h_out

    # Main loop: alternate buffers 0 / 1
    # Main loop: alternate buffers 0 / 1
    start = 0
    batch_id = 0
    while start < num_samples:
        end = min(start + B, num_samples)
        this_bs = end - start
        buf = batch_id & 1  # 0/1

        # If this buffer has a pending result: wait -> write back to all_depths
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

        # 1) Fill pinned host input buffer
        t1 = time.time()
        np.copyto(h_in[buf][:this_bs, :], masks_flat[start:end, :])

        # 2) Async H2D -> record event (event.record on one stream, then event.wait on the other)
        d_masks[buf][:this_bs, :].copy_to_device(h_in[buf][:this_bs, :], stream=stream_h2d)
        ev_h2d[buf].record(stream_h2d)
        gpu_h2d_time += (time.time() - t1)

        # 3) Compute stream waits for H2D -> launch kernel (grid=this_bs)
        ev_h2d[buf].wait(stream_k)  # key: use event.wait(stream)
        blockspergrid = this_bs
        t2 = time.time()
        depth_reduce_kernel[blockspergrid, threadsperblock, stream_k](
            d_masks[buf][:this_bs, :], d_center, area_center, d_out[buf][:this_bs]
        )

        # 4) Async D2H to pinned host output buffer, and record a "done" event on compute stream
        d_out[buf][:this_bs].copy_to_host(h_out[buf][:this_bs], stream=stream_k)
        if done_evt[buf] is None:
            done_evt[buf] = cuda.event(timing=False)
        done_evt[buf].record(stream_k)
        done_slice[buf] = (start, this_bs)

        # Stats
        gpu_compute_time += (time.time() - t2)

        # Next batch
        start   = end
        batch_id += 1


    # Flush remaining results
    for buf in (0, 1):
        if done_evt[buf] is not None:
            t_d2h_wait = time.time()
            done_evt[buf].synchronize()
            gpu_d2h_time += (time.time() - t_d2h_wait)
            s0, ln = done_slice[buf]
            t_h2h = time.time()
            all_depths[s0:s0+ln] = h_out[buf][:ln]
            h2h_copy_time += (time.time() - t_h2h)
            done_evt[buf] = None
            done_slice[buf] = None

    t_total = time.time() - t_total0
    other_time = t_total - preprocessing_time - prewarm_time - gpu_compute_time - gpu_h2d_time - gpu_d2h_time

    # Further break down "other overhead"
    other_setup_alloc_copy = setup_streams_time + alloc_buffers_time + h2h_copy_time
    misc_time = other_time - other_setup_alloc_copy
    if misc_time < 0:
        misc_time = 0.0

    print("\n=== GPU Performance Breakdown (Double Buffering) ===")
    print(f" Total runtime: {t_total:.3f} s")
    print(f"   Preprocessing: {preprocessing_time:.3f} s ({preprocessing_time/t_total*100:.1f}%)")
    print(f"   GPU compute (incl. D2H queued): {gpu_compute_time:.3f} s ({gpu_compute_time/t_total*100:.1f}%)")
    print(f"   H2D transfer (accum.): {gpu_h2d_time:.3f} s ({gpu_h2d_time/t_total*100:.1f}%)")
    print(f"   Pre-warm: {prewarm_time:.3f} s ({prewarm_time/t_total*100:.1f}%)")
    print(f"   D2H wait (accum.): {gpu_d2h_time:.3f} s ({gpu_d2h_time/t_total*100:.1f}%)")
    print(f"   Other overhead: {other_time:.3f} s")
    print("     ├─ Stream/event creation: {:.3f} s".format(setup_streams_time))
    print("     ├─ Device/pinned allocation: {:.3f} s".format(alloc_buffers_time))
    print("     ├─ Host→Host result copy: {:.3f} s".format(h2h_copy_time))
    print("     └─ Misc: {:.3f} s".format(misc_time))
    return all_depths


# =========================
# Data loading / analysis / visualization
# =========================

def load_all_masks(mask_dir):
    """
    Load all mask files in a directory.
    Returns: masks_data, mask_names, affine, header
    """
    print(f"\n=== Loading mask files ===")
    start_time = time.time()

    nii_files = glob.glob(os.path.join(mask_dir, "*.nii"))
    if not nii_files:
        print(f"No .nii files found in directory: {mask_dir}")
        return None, None, None, None

    print(f"Found {len(nii_files)} .nii files")

    first_img = nib.load(nii_files[0])
    first_data = first_img.get_fdata()
    mask_shape = first_data.shape
    affine = first_img.affine
    header = first_img.header

    print(f"Mask shape: {mask_shape}")

    masks_data = np.zeros((len(nii_files),) + mask_shape, dtype=np.float32)
    mask_names = []

    successful_loads = 0
    for i, nii_file in enumerate(nii_files):
        try:
            img = nib.load(nii_file)
            data = img.get_fdata()
            if data.shape != mask_shape:
                print(f"Warning: shape mismatch for {os.path.basename(nii_file)}, skipping")
                continue
            data_f32 = data.astype(np.float32, copy=False)
            masks_data[successful_loads] = data_f32
            mask_names.append(os.path.basename(nii_file))
            successful_loads += 1
        except Exception as e:
            print(f"Load failed: {os.path.basename(nii_file)} - {str(e)}")
            continue

    if successful_loads < len(nii_files):
        masks_data = masks_data[:successful_loads]

    end_time = time.time()
    print(f"Mask loading done, time: {end_time - start_time:.2f} s")
    print(f"Final data shape: {masks_data.shape} ({successful_loads} files)")

    return masks_data, mask_names, affine, header


def analyze_depth_ranking(depth_scores, mask_names):
    print(f"\n=== Depth ranking analysis ===")
    sorted_indices = np.argsort(depth_scores)[::-1]

    print(f"Ranking:")
    print(f"Top 5 masks by depth:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        print(f"  {i+1}. {mask_names[idx]} (depth: {depth_scores[idx]:.4f})")

    print(f"\nBottom 5 masks by depth:")
    start_idx = max(0, len(sorted_indices)-5)
    for i in range(start_idx, len(sorted_indices)):
        idx = sorted_indices[i]
        rank = len(sorted_indices) - i
        print(f"  Bottom {rank}. {mask_names[idx]} (depth: {depth_scores[idx]:.4f})")

    return sorted_indices




def extract_largest_connected_component(binary_data):
    """
    Extract the largest connected component from a binary 3D array.
    All other components are set to 0.
    
    
    Returns:
    largest_component: numpy array, data containing only the largest component
    """
    # Connected-component labeling via scipy.ndimage.label
    # structure defines connectivity (6/18/26); use 26-connectivity (faces/edges/corners)
    structure = np.ones((3, 3, 3), dtype=np.int32)  # 26-connectivity
    labeled_array, num_features = label(binary_data, structure=structure)
    
    if num_features == 0:
        return np.zeros_like(binary_data)
    
    # Compute size of each component
    component_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        component_sizes.append((i, size))
    
    # Find the largest component
    largest_label, largest_size = max(component_sizes, key=lambda x: x[1])
    
    # Keep only the largest component
    largest_component = (labeled_array == largest_label).astype(np.uint8)
    
    return largest_component


def create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir):
    """
    Create three outputs: top-50% depth union, all union, top-50% depth intersection.
    Before operations, binarize masks using threshold 0.5.
    
    Parameters:
    masks_data: numpy array, all mask data
    sorted_indices: numpy array, indices sorted by depth (descending)
    mask_names: list, mask file names
    affine: numpy array, NIfTI affine transform
    header: NIfTI header
    output_dir: str, output directory
    
    Returns:
    results: dict, results for the three operations
    """
    print(f"\n=== Creating mask-operation outputs ===")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Binarize using threshold 0.5
    print(f"\n--- Binarizing with threshold 0.5 ---")
    binary_masks_data = (masks_data >= 0.5).astype(np.float32)
    
    # Save a few binarized masks as reference
    binary_masks_dir = os.path.join(output_dir, "binary_masks")
    os.makedirs(binary_masks_dir, exist_ok=True)
    
    print(f"\n--- Saving first 5 binarized masks as reference ---")
    for i in range(min(5, len(mask_names))):
        binary_mask = binary_masks_data[i]
        binary_mask_uint8 = binary_mask.astype(np.uint8)
        binary_img = nib.Nifti1Image(binary_mask_uint8, affine, header)
        binary_path = os.path.join(binary_masks_dir, f"binary_{mask_names[i]}")
        nib.save(binary_img, binary_path)
    print(f"Saved {min(5, len(mask_names))} binarized masks")
    
    total_count = len(sorted_indices)
    top_50_count = total_count // 2
    
    print(f"\nTotal masks: {total_count}")
    print(f"Top-50% count: {top_50_count}")
    
    # 1) Union of top-50% by depth (binarized)
    print(f"\n--- Computing union of top-50% by depth (binarized) ---")
    top_50_indices = sorted_indices[:top_50_count]
    top_50_binary_masks = binary_masks_data[top_50_indices]
    union_50 = np.any(top_50_binary_masks > 0, axis=0).astype(np.uint8)
    
    # 2) Union of all masks (binarized)
    print(f"\n--- Computing union of all masks (binarized) ---")
    union_100 = np.any(binary_masks_data > 0, axis=0).astype(np.uint8)
    
    # 3) Intersection of top-50% by depth (binarized)
    print(f"\n--- Computing intersection of top-50% by depth (binarized) ---")
    intersection_50 = np.all(top_50_binary_masks > 0, axis=0).astype(np.uint8)
    
    # 4) The deepest single member
    deepest_idx = sorted_indices[0]
    deepest_binary_mask = binary_masks_data[deepest_idx]
    deepest_mask_uint8 = deepest_binary_mask.astype(np.uint8)
    
    # Save files
    results = {}
    
    # ========== Connected-component filtering & saving ==========
    print(f"\n--- Saving largest connected-component results ---")
    
    # 1) Largest component for top-50% union
    union_50_largest = extract_largest_connected_component(union_50)
    
    # 2) Largest component for all union
    union_100_largest = extract_largest_connected_component(union_100)
    
    # 3) Largest component for top-50% intersection
    intersection_50_largest = extract_largest_connected_component(intersection_50)
    
    # Save top-50% union
    union_50_largest_img = nib.Nifti1Image(union_50_largest, affine, header)
    union_50_largest_path = os.path.join(output_dir, "top50percent_depth_union.nii")
    nib.save(union_50_largest_img, union_50_largest_path)
    results['union_50'] = {'data': union_50_largest, 'path': union_50_largest_path}
    print(f"Saved top-50% depth union: {union_50_largest_path}")
    
    # Save all union
    union_100_largest_img = nib.Nifti1Image(union_100_largest, affine, header)
    union_100_largest_path = os.path.join(output_dir, "all_masks_union.nii")
    nib.save(union_100_largest_img, union_100_largest_path)
    results['union_100'] = {'data': union_100_largest, 'path': union_100_largest_path}
    print(f"Saved all-masks union: {union_100_largest_path}")
    
    # Save top-50% intersection
    intersection_50_largest_img = nib.Nifti1Image(intersection_50_largest, affine, header)
    intersection_50_largest_path = os.path.join(output_dir, "top50percent_depth_intersection.nii")
    nib.save(intersection_50_largest_img, intersection_50_largest_path)
    results['intersection_50'] = {'data': intersection_50_largest, 'path': intersection_50_largest_path}
    print(f"Saved top-50% depth intersection: {intersection_50_largest_path}")
    
    # 5) Save the deepest member
    print(f"\n--- Saving deepest member ---")
    print(f"Deepest member: {mask_names[deepest_idx]}")
    
    # Largest component for the deepest member
    deepest_largest = extract_largest_connected_component(deepest_mask_uint8)
    
    # Save deepest member
    deepest_largest_img = nib.Nifti1Image(deepest_largest, affine, header)
    deepest_largest_path = os.path.join(output_dir, f"deepest_member_{mask_names[deepest_idx]}.nii")
    nib.save(deepest_largest_img, deepest_largest_path)
    results['deepest_member'] = {'data': deepest_largest, 'path': deepest_largest_path, 'name': mask_names[deepest_idx]}
    print(f"Saved deepest member: {deepest_largest_path}")
    
    end_time = time.time()
    print(f"Mask operations completed, time: {end_time - start_time:.2f} s")
    
    return results


def visualize_mask_operations(results, output_dir=None):
    """
    Visualize results of the three mask operations.
    
    Parameters:
    results: dict, results for the operations
    output_dir: str, output directory (optional)
    """
    print(f"\n=== Visualizing mask-operation results ===")
    
    fig = plt.figure(figsize=(20, 15))
    
    operations = [
        ('union_50', 'Top 50% Depth Union', 'Reds'),
        ('union_100', 'All Masks Union', 'Blues'), 
        ('intersection_50', 'Top 50% Depth Intersection', 'Greens')
    ]
    
    for op_idx, (op_key, op_title, colormap) in enumerate(operations):
        if op_key not in results:
            continue
            
        mask_data = results[op_key]['data']
        voxel_count = np.sum(mask_data)  # compute directly
        
        # Center slices
        depth, height, width = mask_data.shape
        center_z = depth // 2
        center_y = height // 2
        center_x = width // 2
        
        # Axial slice
        plt.subplot(3, 4, op_idx*4 + 1)
        axial_slice = mask_data[center_z, :, :]
        plt.imshow(axial_slice, cmap=colormap, origin='lower')
        plt.title(f'{op_title}\nAxial (Z={center_z})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        
        # Coronal slice
        plt.subplot(3, 4, op_idx*4 + 2)
        coronal_slice = mask_data[:, center_y, :]
        plt.imshow(coronal_slice, cmap=colormap, origin='lower')
        plt.title(f'Coronal (Y={center_y})')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.colorbar()
        
        # Sagittal slice
        plt.subplot(3, 4, op_idx*4 + 3)
        sagittal_slice = mask_data[:, :, center_x]
        plt.imshow(sagittal_slice, cmap=colormap, origin='lower')
        plt.title(f'Sagittal (X={center_x})')
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.colorbar()
        
        # Stats
        plt.subplot(3, 4, op_idx*4 + 4)
        total_voxels = mask_data.size
        activation_ratio = voxel_count / total_voxels * 100
        
        # Activated voxel counts per slice
        axial_activation = np.sum(mask_data, axis=(1, 2))
        max_axial = np.max(axial_activation)
        
        stats_text = f"""Statistics for {op_title}:

Total Voxels: {total_voxels:,}
Active Voxels: {voxel_count:,}
Activation Ratio: {activation_ratio:.2f}%

Max Axial Activation: {max_axial:,}
Mean Axial Activation: {np.mean(axial_activation):.1f}

Shape: {mask_data.shape}"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title(f'{op_title} - Statistics')
    
    plt.tight_layout()
    plt.show()


# =========================
# Main flow
# =========================

def main():
    print("=== TC Depth Analysis Tool (numba.cuda version) ===")
    total_start_time = time.time()

    preprocessing_time = 0.0
    gpu_compute_time = 0.0
    data_transfer_time = 0.0
    other_overhead_time = 0.0

    # Output directory (relative): TC_DepthAnalysis under the same folder as this .py
    base_dir = Path(__file__).resolve().parent
    output_dir = str(base_dir / "TC_DepthAnalysis")

    print(f"Data source: TC files in Segmentdata")
    print(f"Output directory: {output_dir}")

    # 1) Load TC files as mask data
    t0 = time.time()
    masks_data, mask_names, affine, header = load_TC_files_as_masks()
    # Ensure float32 for probabilistic_inclusion_depth
    if masks_data is not None:
        masks_data = masks_data.astype(np.float32, copy=False)
    preprocessing_time += time.time() - t0

    if masks_data is None:
        print("No TC files were loaded successfully")
        return
    def pick_batch_size(V, bytes_per_voxel=1, safety=0.85, max_batch=1024):
        free_mem, total_mem = cuda.current_context().get_memory_info()
        avail = int(free_mem * safety)
        # Need two input buffers (double buffering) + center (≈ V*4); ignore small output
        overhead = V * 4
        per_batch = V * bytes_per_voxel
        if avail <= overhead + per_batch:
            return 32  # conservative fallback
        max_by_mem = (avail - overhead) // (2 * per_batch)
        B = int(max(1, min(max_by_mem, max_batch)))
        # Align to a multiple of 64
        return max(64, (B // 64) * 64)

    # Before entering GPU branch:
    V = np.prod(masks_data.shape[1:], dtype=np.int64)
    #B = pick_batch_size(V, bytes_per_voxel=4)   # float32: 4 bytes per voxel
    B = 32   # fixed batch size
    # Use B as batch_size
    print(f"B: {B}")
    # 2) Compute
    want_gpu = True  # set True to use GPU
    t0 = time.time()
    depth_scores = probabilistic_inclusion_depth(
        masks_data,
        use_gpu=want_gpu,
        batch_size=B
    )
    gpu_compute_time = time.time() - t0  # rough timing (includes internal preprocessing/transfer)

    # 3) Ranking analysis
    t0 = time.time()
    sorted_indices = analyze_depth_ranking(depth_scores, mask_names)
    other_overhead_time += time.time() - t0

    # 4) Visualization
    t0 = time.time()
    # Only show plots (do not save) unless explicitly requested
    other_overhead_time += time.time() - t0

    # 5) Create mask-operation outputs
    t0 = time.time()
    results = create_mask_operations(masks_data, sorted_indices, mask_names, affine, header, output_dir)
    other_overhead_time += time.time() - t0

    # 6) Visualize mask-operation results
    if results:
        t0 = time.time()
        visualize_mask_operations(results, output_dir=None)
        other_overhead_time += time.time() - t0

    # 7) Save outputs
    print(f"\n=== Saving analysis outputs ===")
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "depth_scores.npy"), depth_scores)
    np.save(os.path.join(output_dir, "sorted_indices.npy"), sorted_indices)
    data_transfer_time = time.time() - t0
    print(f"PID analysis outputs saved to: {output_dir}")

    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    gpu_utilization = (gpu_compute_time / total_runtime) * 100 if total_runtime > 0 else 0.0

    print(f"\n=== Performance Summary (Overall) ===")
    print(f" Total runtime: {total_runtime:.3f} s")
    print(f"   Preprocessing: {preprocessing_time:.3f} s ({preprocessing_time/total_runtime*100:.1f}%)")
    print(f"   GPU compute (incl. internal): {gpu_compute_time:.3f} s ({gpu_compute_time/total_runtime*100:.1f}%)")
    print(f"   Saving time: {data_transfer_time:.3f} s ({data_transfer_time/total_runtime*100:.1f}%)")
    print(f"   Other overhead: {other_overhead_time:.3f} s")
    print(f" Approx. GPU utilization: {gpu_utilization:.1f}%")

    print(f"\n=== Summary ===")
    print(f"Processed {len(mask_names)} mask files")
    print(f"Depth range: {np.min(depth_scores):.4f} - {np.max(depth_scores):.4f}")
    print(f"Mean depth: {np.mean(depth_scores):.4f}")
    
    if results:
        print(f"\nGenerated files:")
        for key, result in results.items():
            if key == 'deepest_member':
                print(f"- {os.path.basename(result['path'])} (deepest: {result['name']})")
            else:
                print(f"- {os.path.basename(result['path'])}")
        print(f"All results saved to: {output_dir}")


def load_TC_files_as_masks():
    """
    Load TC files and convert to the same format as load_all_masks.
    Returns: masks_data, mask_names, affine, header
    """
    print(f"\n=== Loading TC files as mask data ===")
    start_time = time.time()
    
    # First, extract TC files
    tc_data = load_TC_files_as_arrays()
    
    if not tc_data:
        print("No TC files found")
        return None, None, None, None
    
    # Use the first file as reference
    first_folder = list(tc_data.keys())[0]
    first_data_info = tc_data[first_folder]
    mask_shape = first_data_info['shape']
    affine = first_data_info['affine']
    header = first_data_info['header']
    
    print(f"TC mask shape: {mask_shape}")
    
    # Create masks_data array
    masks_data = np.zeros((len(tc_data),) + mask_shape, dtype=np.float32)
    mask_names = []

    successful_loads = 0
    for i, (folder_name, data_info) in enumerate(tc_data.items()):
        try:
            data = data_info['data']
            
            if data.shape != mask_shape:
                print(f"Warning: shape mismatch for {folder_name}, skipping")
                continue
                
            # Convert to float32
            data_f32 = data.astype(np.float32, copy=False)
            
            masks_data[successful_loads] = data_f32
            mask_names.append(f"{folder_name}_TC")
            successful_loads += 1
            
        except Exception as e:
            print(f"Processing failed: {folder_name} - {str(e)}")
            continue
    
    if successful_loads < len(tc_data):
        masks_data = masks_data[:successful_loads]
    
    end_time = time.time()
    print(f"TC loading done, time: {end_time - start_time:.2f} s")
    print(f"Final data shape: {masks_data.shape} ({successful_loads} files)")
    
    return masks_data, mask_names, affine, header

def extract_TC_files_from_soft_masks():
    """
    Extract .nii.gz files whose names contain 'TC' from Segmentdata (same directory as this .py).
    """
    base_dir = Path(__file__).resolve().parent
    source_dir = str(base_dir / "Segmentdata")
    
    print("=== TC File Extractor ===")
    print(f"Source directory: {source_dir}")
    print("-" * 60)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: source directory does not exist: {source_dir}")
        return []
    
    tc_files = []
    
    # Traverse subfolders
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        
        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Find .nii.gz files containing 'TC'
            tc_pattern = os.path.join(folder_path, "*TC*.nii.gz")
            tc_files_in_folder = glob.glob(tc_pattern)
            tc_files.extend(tc_files_in_folder)
    
    print(f"Scan complete! Found {len(tc_files)} TC files")

    return tc_files

def load_TC_files_as_arrays(tc_files_list=None):
    """
    Load TC files into numpy arrays for downstream processing.
    
    Args:
        tc_files_list: list of TC file paths; if None, auto-extract
    
    Returns:
        dict: mapping folder name -> metadata dict (data/affine/header/etc.)
    """
    if tc_files_list is None:
        tc_files_list = extract_TC_files_from_soft_masks()

    if not tc_files_list:
        print("No TC files found!")
        return {}
    
    print(f"\n=== Loading {len(tc_files_list)} TC files into numpy arrays ===")
    tc_data = {}

    for i, file_path in enumerate(tc_files_list, 1):
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        try:
            # Load NIfTI (.nii.gz) via nibabel
            nii_img = nib.load(file_path)
            nii_data = nii_img.get_fdata()
            
            # Store data and metadata
            tc_data[folder_name] = {
                'data': nii_data,
                'affine': nii_img.affine,
                'header': nii_img.header,
                'file_path': file_path,
                'shape': nii_data.shape,
                'dtype': nii_data.dtype,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"Load failed: {os.path.basename(file_path)} - {e}")
    
    print(f"Successfully loaded {len(tc_data)} TC files")
    
    return tc_data

if __name__ == "__main__":
    # Run depth analysis using TC files
    main()
