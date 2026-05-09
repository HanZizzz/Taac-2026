# Docker 运行指南

本项目当前的 [Dockerfile](Dockerfile) 仅包含占位入口（`top -b`），无法直接运行训练。以下给出一种完整且更快的方案（基于 Conda 环境复刻，优先推荐）。

## 一体化方案：使用 Conda 复刻本地环境

### 第一步：准备依赖文件

- 如果你有 `taac_env.yml`，优先使用它。
- 如果没有，也可以保留 `requirements.txt` 作为备选。

`taac_env.yml` 生成方式（在本地终端执行）：

```bash
conda activate taac
conda env export > taac_env.yml
```

### 第二步：编写专用 Dockerfile

在项目根目录创建新的 `Dockerfile`（或例如 `Dockerfile.conda`），内容如下：

```dockerfile
# 1. 直接使用 Miniconda3 作为基础镜像
FROM continuumio/miniconda3:latest

# 2. 设置工作目录
WORKDIR /app

# 3. 配置 Conda 清华源，加速下载
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes

# 4. 复制项目文件
COPY . .

# 5. 如果有 taac_env.yml，直接创建环境（推荐）
# COPY taac_env.yml .
# RUN conda env create -f taac_env.yml && conda clean -afy

# 6. 备选方案：用 Conda 预装常见包，再用 pip 安装其余依赖
RUN conda install -y python=3.10 numpy pandas scipy scikit-learn matplotlib && \
    conda clean -afy && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 默认执行 run.sh
CMD ["bash", "-lc", "./run.sh"]
```

### 第三步：构建并运行

构建镜像（确保 `taac_env.yml` 或 `requirements.txt` 和 Dockerfile 在根目录）：

```bash
docker build -t taac-project:v1 .
```

运行训练（挂载与共享内存设置）：

```bash
# docker run --rm -it --gpus all --shm-size=8g \
#   -v "${PWD}/industrial_data:/data_ams/industrial_training_data" \
#   -e TRAIN_DATA_PATH=/data_ams/industrial_training_data \
#   taac-project:v1

docker run --rm -it --gpus all --shm-size=8g \
  -v "${PWD}/industrial_data:/data_ams/industrial_training_data" \
  -e TRAIN_DATA_PATH=/data_ams/industrial_training_data \
  taac-project:v1 \
  exp_deepfm_din 256 64 0.2 1e-4 0.05
```

需要交互式调试时：

```bash
docker run -it --rm --entrypoint /bin/bash taac-project:v1
```

## 关键说明

- 默认入口为 `run.sh`。
- 环境变量优先级高于 CLI 参数（详见 [train.py](train.py)）。
- 数据规范以 [industrial_data/README.md](industrial_data/README.md) 为准。
