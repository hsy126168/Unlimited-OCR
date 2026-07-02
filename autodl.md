# github hf镜像加速
source /etc/network_turbo
# pip取消加速
unset http_proxy && unset https_proxy

# 驱动，cuda cudnn版本
nvidia-smi
ldconfig -p | grep cuda
ldconfig -p | grep cudnn

# conda、vllm环境
conda activate env
source ~/vllm_env/bin/activate

# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
<!-- pip install --upgrade uv -->
export PATH="/root/.local/bin:$PATH"

# 清理缓存，重试pip
pip cache purge

# 数据盘/root/autodl-tmp
模型与数据放入数据盘，跟账号走，更换实例不影响
cp -r ./models/ /root/autodl-tmp/
ls -la /root/autodl-tmp/models/
ln -s /root/autodl-tmp/models/ ./models

