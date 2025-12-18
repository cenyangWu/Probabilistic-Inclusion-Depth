import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.draw import ellipse, polygon2mask
from scipy.stats import pearsonr, kendalltau, entropy
import pandas as pd

# Helper functions
def _get_agg_axes(masks):
    """Return aggregation axes for summing across spatial dimensions of masks."""
    masks = np.array(masks)
    if masks.ndim == 3:  # (num_masks, height, width)
        return (1, 2)
    elif masks.ndim == 4:  # (num_masks, depth, height, width)
        return (1, 2, 3)
    else:
        raise ValueError(f"Unsupported mask dimensions: {masks.ndim}")

def compute_fuzzy_dice(u, v):
    """Compute the fuzzy Dice coefficient between two (binary/probabilistic) masks."""
    intersection = np.minimum(u, v).sum()
    sum_areas = u.sum() + v.sum()
    if sum_areas == 0:
        return 0.0
    return (2 * intersection) / sum_areas

def compute_epsilon_inclusion_depth(masks) -> np.array: 
    masks = np.array(masks)
    agg_axes = _get_agg_axes(masks)

    inverted_masks = 1 - masks
    area_normalized_masks = (masks.T / np.sum(masks, axis=agg_axes).T).T
    precompute_in = np.sum(inverted_masks, axis=0)
    precompute_out = np.sum(area_normalized_masks, axis=0)

    num_masks = len(masks)
    IN_in = num_masks - np.sum(area_normalized_masks * precompute_in, axis=agg_axes)
    IN_out = num_masks - np.sum(inverted_masks * precompute_out, axis=agg_axes)
    # We remove from the count in_ci, which we do not consider as it adds to both IN_in and IN_out equally
    return (np.minimum(IN_in, IN_out) - 1) / len(masks)

# Define necessary functions
def get_base_gp(num_masks, domain_points, scale=0.01, sigma=1.0, seed=None):
    rng = np.random.default_rng(seed)
    thetas = domain_points.flatten().reshape(-1, 1)
    num_vertices = thetas.size
    gp_mean = np.zeros(num_vertices)

    gp_cov_sin = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.sin(thetas), np.sin(thetas), "sqeuclidean"))
    gp_sample_sin = rng.multivariate_normal(gp_mean, gp_cov_sin, num_masks)
    gp_cov_cos = scale * np.exp(-(1 / (2 * sigma)) * cdist(np.cos(thetas), np.cos(thetas), "sqeuclidean"))
    gp_sample_cos = rng.multivariate_normal(gp_mean, gp_cov_cos, num_masks)

    return gp_sample_sin + gp_sample_cos

def get_xy_coords(angles, radii):
    num_members = radii.shape[0]
    angles = angles.flatten().reshape(1, -1)
    angles = np.repeat(angles, num_members, axis=0)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def rasterize_coords(x_coords, y_coords, num_rows, num_cols):
    masks = []
    for xc, yc in zip(x_coords, y_coords):
        coords_arr = np.vstack((xc, yc)).T
        coords_arr *= num_rows // 2
        coords_arr += num_cols // 2
        mask = polygon2mask((num_rows, num_cols), coords_arr).astype(float)
        masks.append(mask)
    return masks

def main_shape_with_outliers(num_masks, num_rows, num_cols, num_vertices=100, 
                             population_radius=0.5,
                             normal_scale=0.003, normal_freq=0.9,
                             outlier_scale=0.009, outlier_freq=0.04,
                             p_contamination=0.5, return_labels=False, seed=None):

    rng = np.random.default_rng(seed)
    thetas = np.linspace(0, 2 * np.pi, num_vertices)
    population_radius = np.ones_like(thetas) * population_radius

    gp_sample_normal = get_base_gp(num_masks, thetas, scale=normal_scale, sigma=normal_freq, seed=seed)+0.1
    gp_sample_outliers = get_base_gp(num_masks, thetas, scale=outlier_scale, sigma=outlier_freq, seed=seed)

    should_contaminate = rng.random(num_masks) < p_contamination
    should_contaminate = should_contaminate.reshape(-1, 1)
    should_contaminate = np.repeat(should_contaminate, len(thetas), axis=1)

    radii = population_radius + gp_sample_normal * (~should_contaminate) + gp_sample_outliers * should_contaminate

    xs, ys = get_xy_coords(thetas, radii)
    contours = rasterize_coords(xs, ys, num_rows, num_cols)
    labels = should_contaminate[:, 0].astype(int)

    if return_labels:
        return contours, labels
    else:
        return contours

def compute_sdf(binary_mask):
    inside_dist = distance_transform_edt(binary_mask)
    outside_dist = distance_transform_edt(1 - binary_mask)
    sdf = inside_dist - outside_dist
    return sdf

def compute_mutual_information(sdf1, sdf2, bins=128):
    """
    (Robust version) Compute the normalized mutual information (NMI) between two 2D SDFs.
    Includes NaN checks for stability.
    """
    try:
        x = sdf1.flatten()
        y = sdf2.flatten()
        
        # 1) Estimate joint distribution
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        
        # 2) Marginals
        px = np.sum(joint_hist, axis=1)
        py = np.sum(joint_hist, axis=0)
        
        # 3) Filter zeros
        px_nz = px[px > 0]
        py_nz = py[py > 0]
        joint_hist_nz = joint_hist[joint_hist > 0]

        # --- Robustness checks ---
        # If histograms are all zeros (e.g., constant SDF), entropy is 0.
        if len(px_nz) == 0 or len(py_nz) == 0 or len(joint_hist_nz) == 0:
            # If both SDFs are constant, treat as perfectly correlated.
            return 1.0 

        # 4) Entropies
        H_X = entropy(px_nz, base=2)
        H_Y = entropy(py_nz, base=2)
        H_XY = entropy(joint_hist_nz, base=2)
        
        # --- Robustness checks (avoid NaN) ---
        if np.isnan(H_X) or np.isnan(H_Y) or np.isnan(H_XY):
            print("Warning: encountered NaN while computing entropy.")
            return 0.0  # Return a valid number instead of NaN

        # 5) Mutual information
        I_XY = H_X + H_Y - H_XY
        
        # 6) NMI
        sum_H = H_X + H_Y
        if sum_H == 0:
            # If I_XY is also 0 (since H_X, H_Y, H_XY are all 0), return 1.0
            return 1.0 if I_XY == 0 else 0.0

        NMI = (2.0 * I_XY) / sum_H
        
        # Final check
        if np.isnan(NMI):
            print(f"Warning: final NMI is NaN. I_XY={I_XY}, sum_H={sum_H}")
            return 0.0
            
        return NMI
    
    except Exception as e:
        print(f"Error while computing mutual information: {e}")
        return 0.0
# Generate contours
num_samples = 200
num_rows = num_cols = 100
size_window = 100
contours_masks, _ = main_shape_with_outliers(num_samples, num_rows, num_cols, return_labels=True, seed=6)

window = np.zeros((num_samples, size_window, size_window), dtype=np.float32)
i = 0
j = 0
for k in range(num_samples):
    window[k] = contours_masks[k][i:i+size_window, j:j+size_window]

def probabilistic_inclusion_depth_cpu(masks_data):
    """
    Compute Probabilistic Inclusion Depth (PID) on CPU.

    Uses the identity:
    1 - sum(M*(1-C))/sum(M) == sum(M*C)/sum(M)
    """
    num_samples = masks_data.shape[0]
    # 1) Flatten
    masks_flattened = masks_data.reshape(num_samples, -1)
    # 2) Center
    original_center = np.mean(masks_flattened, axis=0, dtype=np.float32)
    # 3) Denominators (areas)
    area_masks = np.sum(masks_flattened, axis=1, dtype=np.float32)
    area_center = np.sum(original_center, dtype=np.float32)
    # 4) Numerator (shared dot product)
    dot_product = np.sum(masks_flattened * original_center, axis=1, dtype=np.float32)
    # 5) Compute s1 and s2
    # s1 = (M·C) / |M|
    s1 = np.divide(dot_product, area_masks, out=np.zeros_like(dot_product), where=area_masks!=0)
    # s2 = (M·C) / |C|
    if area_center > 0:
        s2 = dot_product / area_center
    else:
        s2 = np.zeros_like(dot_product)
        
    # 6) Depth is the minimum of the two
    depths = np.minimum(s1, s2)
    
    return depths.astype(np.float32, copy=False)

# Compute PID depth
print("Computing PID depth...")

depth_scores = probabilistic_inclusion_depth_cpu(window.reshape(num_samples, size_window, size_window))

# Normalize PID scores (higher is better; normalized values are used for sorting)
min_inclusion = depth_scores.min()
max_inclusion = depth_scores.max()
if max_inclusion > min_inclusion:
    normalized_inclusion_scores = (depth_scores - min_inclusion) / (max_inclusion - min_inclusion)
else:
    normalized_inclusion_scores = depth_scores


def compute_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    for i in range(num_masks):
        inclusion_mat[i, :] = np.all((masks & masks[i]) == masks[i], axis=(1, 2))
        inclusion_mat[i, i] = 0
    return inclusion_mat


def compute_epsilon_inclusion_matrix(masks):
    """Matrix that, per contour says if its inside (1) or outside (-1).
    The entry is 0 if the relationship is ambiguous. 

    Parameters
    ----------
    masks : list
        list of ndarrays corresponding to an ensemble of binary masks.
    """
    num_masks = len(masks)
    masks = np.array(masks).astype(int)
    inclusion_mat = np.zeros((num_masks, num_masks))
    inv_masks = 1 - masks
    area = np.sum(masks, axis=(1, 2))
    
    for i in range(num_masks):
        if area[i] > 0:  # Check to prevent division by zero
            inclusion_scores = 1 - np.sum(inv_masks & masks[i], axis=(1, 2)) / area[i]
        else:
            inclusion_scores = np.zeros(num_masks)  # If mask area is 0, set all relations to 0
            
        inclusion_mat[i, :] = inclusion_scores
        inclusion_mat[i, i] = 0  # Set diagonal to 0 as a mask cannot include itself

    return inclusion_mat


def sorted_depth(masks, depth="eid"):
    masks = np.array(masks, dtype=np.float32)
    num_masks = masks.shape[0]
    assert(depth in ["eid", "id", "cbd", "ecbd"])
    if depth == "eid" or depth == "ecbd":
        inclusion_matrix = compute_epsilon_inclusion_matrix(masks)
        np.fill_diagonal(inclusion_matrix, 1)  # Required for feature parity with the O(N) version of eID.
    else:
        inclusion_matrix = compute_inclusion_matrix(masks)
    N = num_masks
    depths = np.zeros(num_masks)  # Store depth values per mask

    for i in range(num_masks):
        if depth in ["cbd", "id", "eid", "ecbd"]:
            N_a = np.sum(inclusion_matrix[i])
            N_b = np.sum(inclusion_matrix.T[i])

        if depth == "cbd" or depth == "ecbd":
            N_ab_range = N
            depth_in_cluster = (N_a * N_b) / (N_ab_range * N_ab_range)
        else:  # ID / eID
            depth_in_cluster = np.minimum(N_a, N_b) / N

        depths[i] = depth_in_cluster  # Store depth for current mask

    # Sort depths (descending)
    sorted_indices = np.argsort(-depths)

    return sorted_indices

# Compute multiple sorting results
print("\nComputing eID(PID) depth...")
# Use eID(PID) depth function
eid_depths = compute_epsilon_inclusion_depth(contours_masks)
id_sorted_indices = np.argsort(-eid_depths)  # Sort by depth (descending)

# Compute Prob-IoU depth (each member vs mean)
print("Computing Prob-IoU depth...")

# Mean mask
mean_mask = np.mean(window, axis=0)

# Fuzzy Dice vs mean (as a proxy score)
fuzzy_iou_scores = []
for i in range(num_samples):
    # Binarize probabilistic masks (threshold 0.5)
    binary_mean = (mean_mask > 0.5).astype(float)
    binary_member = (window[i] > 0.5).astype(float)
    
    fuzzy_iou = compute_fuzzy_dice(binary_member, binary_mean)
    fuzzy_iou_scores.append(fuzzy_iou)

fuzzy_iou_scores = np.array(fuzzy_iou_scores)
fuzzy_iou_sorted_indices = np.argsort(-fuzzy_iou_scores)  # Sort by Prob-IoU (descending)

# Compute ISM depth
print("Computing ISM depth for all members...")

# 1) SDF of the mean mask (compute once)
binary_mean_iso = (mean_mask > 0.5).astype(float)
sdf_mean_iso = compute_sdf(binary_mean_iso)

# 2) Loop over members and compute similarity vs mean
iso_similarity_scores = []
for i in range(num_samples):
    
    # Member i
    binary_member_iso = (window[i] > 0.5).astype(float)
    
    # SDF of member i
    sdf_member_iso = compute_sdf(binary_member_iso)
    
    # Mutual information vs mean (ISM)
    iso_similarity = compute_mutual_information(sdf_mean_iso, sdf_member_iso, bins=128)
    iso_similarity_scores.append(iso_similarity)

# 3) Sort
iso_similarity_scores = np.array(iso_similarity_scores)
iso_similarity_sorted_indices = np.argsort(-iso_similarity_scores)  # Sort by similarity (descending)

# CBD depth
print("Computing CBD depth...")
cbd_sorted_indices = sorted_depth(contours_masks, depth="cbd")  # CBD sorting

original_sorted_indices = np.argsort(-normalized_inclusion_scores)  # PID sorting (descending)


# Analyze differences among five sorting methods
print("\nAnalysis of Five Sorting Methods:")
methods = ['eID(PID)', 'CBD', 'Prob-IoU', 'PID-Mean', 'ISM']
sorted_indices_list = [id_sorted_indices, cbd_sorted_indices, fuzzy_iou_sorted_indices, original_sorted_indices, iso_similarity_sorted_indices]
# Compute ranks under each method
all_ranks = np.zeros((num_samples, 5))
for i, sorted_indices in enumerate(sorted_indices_list):
    for j, idx in enumerate(sorted_indices):
        all_ranks[idx, i] = j

# Create scatterplot matrix (SPLOM) - 5x5
n_methods = len(methods)

# Pearson correlation matrix (upper triangle)
pearson_matrix = np.zeros((n_methods, n_methods))
for i in range(n_methods):
    for j in range(n_methods):
        pearson_corr, _ = pearsonr(all_ranks[:, i], all_ranks[:, j])
        pearson_matrix[i, j] = pearson_corr
fig, axes = plt.subplots(n_methods, n_methods, figsize=(30, 30), 
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

for i in range(n_methods):
    for j in range(n_methods):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: method name
            ax.text(0.5, 0.5, methods[i], transform=ax.transAxes, 
                   fontsize=36, ha='center', va='center', fontweight='bold')
            ax.axis('off')  # Hide axes
        elif i > j:
            # Lower-left: scatter plot
            ax.scatter(all_ranks[:, j], all_ranks[:, i], alpha=0.6, s=50, color='#4A7CB3')
            ax.plot([0, num_samples-1], [0, num_samples-1], 'k--', alpha=0.5, linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')  # Keep square aspect
            
            # Ticks
            ax.locator_params(axis='both', nbins=5)
            
            # Show ticks/labels only on outer edges
            if j == 0:
                # Left edge: show y ticks
                ax.tick_params(axis='y', which='major', labelsize=18, labelleft=True)
            else:
                # Not left edge: hide y tick labels
                ax.tick_params(axis='y', which='major', labelleft=False)
            
            if i == n_methods - 1:
                # Bottom edge: show x ticks
                ax.tick_params(axis='x', which='major', labelsize=18, labelbottom=True)
            else:
                # Not bottom edge: hide x tick labels
                ax.tick_params(axis='x', which='major', labelbottom=False)
        else:  # i < j
            # Upper-right: Pearson correlation
            pearson_val = pearson_matrix[i, j]
            
            # Background intensity based on |correlation|
            color_intensity = abs(pearson_val)
            bg_color = plt.cm.Blues(color_intensity * 0.6 + 0.2)
            
            # Hide axes and set background color
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor(bg_color)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Draw a rectangle background
            rect = Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor='lightgray', linewidth=1, transform=ax.transAxes, zorder=0)
            ax.add_patch(rect)
            
            # Display value
            ax.text(0.5, 0.5, f'{pearson_val:.3f}', transform=ax.transAxes, 
                   fontsize=36, ha='center', va='center', fontweight='bold',
                   color='white' if color_intensity > 0.5 else 'black', zorder=10)
            
            # Light borders
            for spine in ax.spines.values():
                spine.set_edgecolor('lightgray')
                spine.set_linewidth(0.5)

plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for colorbar

# Colorbar on the right
norm = Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(cmap=cm.Blues, norm=norm)
sm.set_array([])

# Add colorbar to the right
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.03, aspect=25)
cbar.set_label('Pearson Correlation Coefficient', fontsize=30, fontweight='bold', labelpad=15)
cbar.ax.tick_params(labelsize=30, pad=10)

SAVE_FIGURES = False  # Do not save figures unless explicitly requested
if SAVE_FIGURES:
    plt.savefig('rank_correlation_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

# Pairwise correlations and Kendall Tau distances among methods
linear_corr_matrix = np.zeros((5, 5))
kendall_corr_matrix = np.zeros((5, 5))
kendall_distance_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        # Correlation on ranks
        linear_corr, _ = pearsonr(all_ranks[:, i], all_ranks[:, j])
        linear_corr_matrix[i, j] = linear_corr
        
        # Kendall Tau
        kendall_corr, _ = kendalltau(all_ranks[:, i], all_ranks[:, j])
        kendall_corr_matrix[i, j] = kendall_corr
        
        # Kendall Tau distance = (1 - tau) / 2
        kendall_distance_matrix[i, j] = (1 - kendall_corr) / 2

# Pretty print as DataFrames
linear_corr_df = pd.DataFrame(linear_corr_matrix, index=methods, columns=methods)
kendall_corr_df = pd.DataFrame(kendall_corr_matrix, index=methods, columns=methods)
kendall_distance_df = pd.DataFrame(kendall_distance_matrix, index=methods, columns=methods)

print("\nPearson linear correlation coefficients:")
print(linear_corr_df)

print("\nKendall Tau coefficients:")
print(kendall_corr_df)

print("\nKendall Tau distances:")
print(kendall_distance_df)

# Shared colorscale configuration
CORRELATION_COLORSCALE = {
    'pearson': {'vmin': -1, 'vmax': 1},
    'kendall_distance': {'vmin': 0, 'vmax': 0.5}
}

# Visualize correlation matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Pearson heatmap
im0 = axes[0].imshow(linear_corr_matrix, cmap='coolwarm', **CORRELATION_COLORSCALE['pearson'])
axes[0].set_title('Pearson', fontsize=18)
axes[0].set_xticks(np.arange(len(methods)))
axes[0].set_yticks(np.arange(len(methods)))
axes[0].set_xticklabels(methods, fontsize=14)
axes[0].set_yticklabels(methods, fontsize=14)
plt.colorbar(im0, ax=axes[0])

# Value labels
for i in range(len(methods)):
    for j in range(len(methods)):
        text = axes[0].text(j, i, f"{linear_corr_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black" if abs(linear_corr_matrix[i, j]) < 0.7 else "white",
                           fontsize=11)

# Kendall Tau distance heatmap
im1 = axes[1].imshow(kendall_distance_matrix, cmap='coolwarm', **CORRELATION_COLORSCALE['kendall_distance'])
axes[1].set_title('Kendall Tau', fontsize=18)
axes[1].set_xticks(np.arange(len(methods)))
axes[1].set_yticks(np.arange(len(methods)))
axes[1].set_xticklabels(methods, fontsize=14)
axes[1].set_yticklabels(methods, fontsize=14)
plt.colorbar(im1, ax=axes[1])

# Value labels
for i in range(len(methods)):
    for j in range(len(methods)):
        text = axes[1].text(j, i, f"{kendall_distance_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black" if kendall_distance_matrix[i, j] > 0.25 else "white",
                           fontsize=11)



