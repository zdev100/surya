from mypagelayout import LayoutPredictor
from PIL import Image, ImageDraw, ImageFont

# 读取指定的图片
img = Image.open('/Users/zworker/dev/AI/surya_zdev100/mytest/output_images2/2505.15734v1_page1.png')

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
try:
    font = ImageFont.truetype("arial.ttf", 96)  # 字号调大，适合高DPI
except Exception:
    print("arial.ttf 字体文件未找到，使用默认字体")
    font = ImageFont.load_default()

for result in results:
    for idx, box in enumerate(result.bboxes):
        # box.polygon 是四个点的列表
        polygon = [tuple(map(int, pt)) for pt in box.polygon]
        draw.polygon(polygon, outline="red", width=3)
        # 标注label
        if hasattr(box, 'label'):
            draw.text(polygon[0], str(box.label), fill="blue", font=font)
        # 标注阅读顺序编号，序号大一些
        draw.text((polygon[0][0], polygon[0][1] + 40), str(idx + 1), fill="green", font=font)

# 保存可视化图片
img_draw.save("mytest/output_images2/visualized_result.png")
img_draw.show()
