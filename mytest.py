from mypagelayout import LayoutPredictor
from PIL import Image, ImageDraw

# 读取指定的图片
img = Image.open('/Users/zworker/dev/AI/surya_zdev100/mytest/output_images2/2505.15734v1_page2.png')

# 实例化预测器
predictor = LayoutPredictor()

# 执行预测
results = predictor([img])

# 打印结果
for result in results:
    print(result)

# 可视化输出
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for result in results:
    for idx, box in enumerate(result.bboxes):
        # box.polygon 是四个点的列表
        polygon = [tuple(map(int, pt)) for pt in box.polygon]
        draw.polygon(polygon, outline="red", width=3)
        # 标注label
        if hasattr(box, 'label'):
            draw.text(polygon[0], str(box.label), fill="blue")
        # 标注阅读顺序编号
        draw.text(polygon[0], str(idx + 1), fill="green")

img_draw.show()
