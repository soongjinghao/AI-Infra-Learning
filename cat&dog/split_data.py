import os
import shutil
import random

# ====== 根据你的实际情况设置路径 ======
# 源路径加上 PetImages 文件夹
src_dir = r"D:\kaggle_data\datasets\shaunthesheep\microsoft-catsvsdogs-dataset\versions\1\PetImages"
# 你的项目 data 目录
dst_dir = r"/cat&dog/data"

# 清理并创建目标目录（防止残留文件干扰）
for split in ['train', 'val']:
    for category in ['cat', 'dog']:
        os.makedirs(os.path.join(dst_dir, split, category), exist_ok=True)

# 固定随机种子，保证每次分配一致
random.seed(42)

# 处理猫和狗
for category in ['cat', 'dog']:
    # 注意：原始文件夹名字首字母大写（Cat, Dog），目标文件夹小写（cat, dog）
    folder_name = 'Cat' if category == 'cat' else 'Dog'
    src_cat_dir = os.path.join(src_dir, folder_name)

    # 获取所有图片
    images = os.listdir(src_cat_dir)
    random.shuffle(images)

    # 取前 1000 张做训练，接下来的 200 张做验证
    train_images = images[:1000]
    val_images = images[1000:1200]

    print(f"正在复制 {category} 的图片...")
    # 复制文件
    for img in train_images:
        shutil.copy(os.path.join(src_cat_dir, img), os.path.join(dst_dir, 'train', category, img))
    for img in val_images:
        shutil.copy(os.path.join(src_cat_dir, img), os.path.join(dst_dir, 'val', category, img))

print("\n数据整理完成！")
print(
    f"训练集：猫 {len(os.listdir(os.path.join(dst_dir, 'train', 'cat')))} 张，狗 {len(os.listdir(os.path.join(dst_dir, 'train', 'dog')))} 张")
print(
    f"验证集：猫 {len(os.listdir(os.path.join(dst_dir, 'val', 'cat')))} 张，狗 {len(os.listdir(os.path.join(dst_dir, 'val', 'dog')))} 张")