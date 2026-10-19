import os
# 1. 设置自定义缓存路径（必须在 import kagglehub 之前）
os.environ["KAGGLEHUB_CACHE"] = r"D:\kaggle_data"

import kagglehub

# 2. 登录验证（如果之前已经登录成功过，它可能会自动跳过；如果没登录过，这里会让你输入Token）
print("正在检查登录状态...")
kagglehub.login()

# 3. 下载数据集（只下载一次！）
dataset_id = "shaunthesheep/microsoft-catsvsdogs-dataset"
print("开始下载数据集，请耐心等待...")
path = kagglehub.dataset_download(dataset_id)
print(f"下载完成！文件保存在: {path}")