import numpy as np
import scipy
import nibabel as nib
import numba
from numba import cuda
import matplotlib
import pandas as pd
import skimage
import sys
from numba import cuda

# 打印驱动版本和运行时版本

print(f"cuda Runtime Version: {cuda.runtime.get_version()}")

# 检测硬件并打印汇总
cuda.detect()
print(f"Python version:    {sys.version.split()[0]}")
print(f"--- 核心第三方库 ---")
print(f"NumPy version:      {np.__version__}")      # 用于矩阵运算
print(f"SciPy version:      {scipy.__version__}")   # 用于统计(stats)和图像处理(ndimage)
print(f"Nibabel version:    {nib.__version__}")     # 用于读取 .nii 医学影像
print(f"Numba version:      {numba.__version__}")   # 用于 @cuda.jit 加速
print(f"Matplotlib version: {matplotlib.__version__}")# 用于绘图
print(f"Pandas version:     {pd.__version__}")     # 用于处理结果表格
print(f"Scikit-image:       {skimage.__version__}") # 用于 polygon2mask 等图像算法

print(f"\n--- GPU 环境检查 ---")
try:
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"CUDA Available:     Yes")
        print(f"GPU Device:         {device.name}")
    else:
        print("CUDA Available:     No (Check your NVIDIA drivers)")
except Exception as e:
    print(f"CUDA Check Error:   {e}")
