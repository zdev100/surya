from mypagelayout import LayoutPredictor
from PIL import Image

# 读取指定的图片
img = Image.open('mytest/output_images/2505.06236v1_page13.png')

# 实例化预测器
predictor = LayoutPredictor()

# 执行预测
results = predictor([img])

# 打印结果
for result in results:
    print(result)
