import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def main():
    # ====== 请在这里替换成你要预测的图片路径 ======
    # 找一张猫或狗的图片，把路径写在下面（注意前面的 r 不要删）
    image_path = r"C:\Users\28789\Desktop\test_cat.jpg"
    # ============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 1. 重建模型结构（必须和训练时一模一样）
    model = models.resnet18(weights=None)  # 这里不需要下载预训练权重了
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 2. 加载刚才保存的模型权重
    model.load_state_dict(torch.load("cats_dogs_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()  # 切换到评估模式，非常重要！

    # 3. 图片预处理（必须和验证集的处理方式一致）
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 4. 读取并处理图片
    try:
        img = Image.open(image_path).convert("RGB")  # 确保是 RGB 格式，防出错
    except FileNotFoundError:
        print(f"找不到图片，请检查路径: {image_path}")
        return

    img_tensor = preprocess(img)
    # 增加一个维度，变成 [1, 3, 224, 224]，因为模型默认接收 batch 数据
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 5. 开始预测
    with torch.no_grad():
        outputs = model(img_tensor)
        # outputs 是 [[猫的概率, 狗的概率]]，取较大值的索引
        _, pred = torch.max(outputs, 1)
        pred_idx = pred.item()

    # 6. 因为 ImageFolder 是按字母顺序排序的，所以 0 是 cat，1 是 dog
    class_names = ["cat", "dog"]
    result = class_names[pred_idx]

    # 打印结果
    print(f"图片路径: {image_path}")
    print(f"模型预测结果: {result}")
    print(f"原始输出概率 (猫, 狗): {torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]}")

if __name__ == "__main__":
    main()