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
# RUN conda install -y python=3.10 numpy pandas scipy scikit-learn matplotlib && \
#     conda clean -afy && \
#     pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simpleRUN conda install -y python=3.10 && \
    

RUN conda install -y python=3.10 && \
    conda clean -afy && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ① 先清除 base 镜像自带的 ENTRYPOINT（防止它拦截 CMD）
ENTRYPOINT []
CMD ["bash", "-lc", "exec ./run.sh --batch_size 256 --d_model 64 --dropout_rate 0.2 --lr 1e-4 --sparse_lr 0.05"]

