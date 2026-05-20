#!/usr/bin/env bash
set -e

echo "===== Installing single-cell benchmark environment ====="
echo "Date: $(date)"
echo "Python: $(python --version)"

# 1. 升级 pip 并配置清华源
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 2. 基础科学计算包
pip install --force-reinstall --no-cache-dir \
  "numpy==1.26.4" \
  "pandas==2.2.2" \
  "scipy==1.12.0" \
  "scikit-learn==1.6.1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 3. 单细胞基础包
pip install --no-cache-dir \
  "scanpy==1.11.5" \
  "anndata==0.11.4" \
  "h5py>=3.10" \
  matplotlib \
  pyyaml \
  ipython \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. scGPT 包
pip install --no-cache-dir scgpt ipython -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 5. 安装 PyTorch 全家桶 (全部指定版本避免兼容性问题)
pip uninstall -y torch torchvision torchaudio torchtext || true

pip install --no-cache-dir \
  torch==2.3.1 \
  torchvision==0.18.1 \
  torchaudio==2.3.1 \
  torchtext==0.18.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/

echo "===== Installation complete ====="
