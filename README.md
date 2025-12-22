# Probabilistic Inclusion Depth for Fuzzy Contour Ensemble Visualization

#### [Project Page](https://github.com/cenyangWu/Probabilistic-Inclusion-Depth) | [arXiv Paper](https://arxiv.org/abs/2512.15187)

Cenyang Wu, Daniel Klötzl, Qinhan Yu, Shudan Guo, Runhao Lin, Daniel Weiskopf, and Liang Zhou†

Welcome to the official repository for **Probabilistic Inclusion Depth (PID)** This project introduces a data depth method for ensemble visualization of scalar fields. By introducing a probabilistic inclusion operator $\subset_p$, our method supports ensembles of fuzzy contours (such as soft masks from modern segmentation methods) and conventional ensembles of binary contours. PID enables contour boxplot visualization for 3D medical imaging, scalar field ensembles, and probabilistic segmentation outputs.

![teaser](figures/teaser.png)


## 🚀 Getting Started

Follow these steps to set up the environment and start using PID for your ensemble visualization tasks.

### 1. Environment Setup

First, clone the repository and set up the conda environment.

```shell
git clone https://github.com/cenyangWu/Probabilistic-Inclusion-Depth.git
cd Probabilistic-Inclusion-Depth
conda create --name pid-env -y python=3.9
conda activate pid-env
pip install --upgrade pip setuptools
```

Next, install the required Python packages with specific versions.

```shell
# Install core scientific computing packages
pip install numpy==1.26.4
pip install scipy==1.12.0
pip install matplotlib==3.8.0
pip install pandas==2.2.0

# Install image processing and medical imaging packages
pip install nibabel==5.3.2
pip install scikit-image==0.21.0

# Install Numba for CPU acceleration and CUDA support (requires CUDA 11.8)
pip install numba==0.60.0

# Verify CUDA is working correctly (optional but recommended)
python -c "from numba import cuda; print('CUDA available:', cuda.is_available())" || echo "WARNING: CUDA not available, will use CPU fallback"
```

**GPU Requirements**: For GPU acceleration, ensure you have:
- NVIDIA GPU with Compute Capability ≥ 7.0 (tested on RTX 4080 Super with CC 8.9)
- CUDA Toolkit 11.8 or compatible version
- Sufficient GPU memory (16GB+ recommended for large 3D ensembles)

### 2. Project Structure

Organize your workspace as follows:

```
Probabilistic-Inclusion-Depth/
 ├─ PID/
 │   ├─ Boxplot.py              # Main PID computation with GPU acceleration
 │   ├─ ScatterplotMatrix.py    # Depth method comparison and ranking analysis
 │   └─ Segmentdata/            # (Optional) Place your segmentation data here
 │       ├─ sample1/
 │       │   └─ *TC*.nii.gz
 │       ├─ sample2/
 │       │   └─ *TC*.nii.gz
 │       └─ ...
 ├─ BoxplotRendering/           # 3D visualization utilities
 │   ├─ render.py
 │   ├─ process_all_nii.py
 │   └─ data/
 ├─ README.md
 └─ ...
```

#### Running Examples

**Brain Tumor Segmentation (Soft Masks)**:
```shell
cd PID
python Boxplot.py
```
This will:
- Load TC (tumor core) segmentation masks from `Segmentdata/`
- Compute PID depth scores using GPU acceleration
- Generate 3D contour boxplots
- Save results to `TC_DepthAnalysis/`

**Depth Method Comparison**:
```shell
python PID/ScatterplotMatrix.py
```
This generates synthetic contour ensembles and compares multiple depth methods:
- **PID-Mean**: Our efficient approximation
- **eID**: Epsilon Inclusion Depth (binary specialization of PID)
- **CBD**: Contour Band Depth
- **Prob-IoU**: Probabilistic IoU-based depth
- **ISM**: Isosurface Similarity Map depth

## ✨ Method Overview

### Probabilistic Inclusion Operator

For two probabilistic masks $u$ and $v$, the probabilistic inclusion operator is defined as:

$$u \subset_p v := \mathbb{E}_{X\sim \pi_u}[v(X)] = \frac{\int u \cdot v \, \mathrm{d}\mu}{\int u \, \mathrm{d}\mu} = 1 - \frac{\int u(1-v) \, \mathrm{d}\mu}{\int u \, \mathrm{d}\mu}$$

### Probabilistic Inclusion Depth (PID)

For an ensemble of $N$ contours with probabilistic masks $\{u_1, u_2, \ldots, u_N\}$:

$$\mathrm{IN}_{\mathrm{in}}^{\mathrm{p}}(u_i) = \frac{1}{N}\sum_{j=1}^{N}(u_i \subset_p u_j), \quad \mathrm{IN}_{\mathrm{out}}^{\mathrm{p}}(u_i) = \frac{1}{N}\sum_{j=1}^{N}(u_j \subset_p u_i)$$

$$\mathrm{PID}(u_i) = \min\\{\mathrm{IN}_{\mathrm{in}}^{\mathrm{p}}(u_i), \mathrm{IN}_{\mathrm{out}}^{\mathrm{p}}(u_i)\\}$$

### PID-Mean (Efficient Linear Approximation)

Instead of pairwise comparisons, PID-mean compares each contour only against the mean contour:

$$\bar{u}(x) = \frac{1}{N}\sum_{i=1}^{N}u_i(x)$$

$$\mathrm{IN}_{\mathrm{in}}^{\mathrm{mean}}(u_i) = u_i \subset_p \bar{u}, \quad \mathrm{IN}_{\mathrm{out}}^{\mathrm{mean}}(u_i) = \bar{u} \subset_p u_i$$

$$\mathrm{PID\text{-}mean}(u_i) = \min\\{\mathrm{IN}_{\mathrm{in}}^{\mathrm{mean}}(u_i), \mathrm{IN}_{\mathrm{out}}^{\mathrm{mean}}(u_i)\\}$$

**Complexity**: $O(MN)$ where $M$ is the number of voxels and $N$ is ensemble size.




## 📖 Citation

If you find PID useful in your research, please consider citing our paper.

```bibtex
@misc{wu2025probabilisticinclusiondepthfuzzy,
      title={Probabilistic Inclusion Depth for Fuzzy Contour Ensemble Visualization}, 
      author={Cenyang Wu and Daniel Klötzl and Qinhan Yu and Shudan Guo and Runhao Lin and Daniel Weiskopf and Liang Zhou},
      year={2025},
      eprint={2512.15187},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2512.15187}, 
}
```