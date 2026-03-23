# 导入所需库
import cv2  # OpenCV 用于图像处理
import numpy as np  # NumPy 用于数值操作
import matplotlib.pyplot as plt  # Matplotlib 用于图像显示

# ===================== 任务1：读取测试图片 =====================
image_path = "/home/lenovo/cv-course/labe01/labe01/src/屏幕截图 2026-01-06 103054.png"  # 示例：测试图片路径
# 读取图片，cv2.imread 默认以 BGR 格式读取彩色图像
img = cv2.imread(image_path)

# 检查图片是否成功读取
if img is None:
    raise ValueError(f"无法读取图片，请检查路径是否正确：{image_path}")

# ===================== 任务2：输出图像基本信息 =====================
# 获取图像尺寸：height(高度/长度)、width(宽度)、channels(通道数)
height, width = img.shape[:2]  # 前两个维度是高和宽
channels = img.shape[2] if len(img.shape) == 3 else 1  # 彩色图3通道，灰度图无第3维度
dtype = img.dtype  # 图像数据类型（通常是uint8，即8位无符号整数）

# 终端打印信息
print("===== 图像基本信息 =====")
print(f"图像宽度：{width} 像素")
print(f"图像高度（长度）：{height} 像素")
print(f"图像通道数：{channels}")
print(f"图像数据类型：{dtype}")
print(f"图像整体尺寸（高, 宽, 通道）：{img.shape}")

# ===================== 任务3：显示原图 =====================
# OpenCV 读取的是 BGR 格式，Matplotlib 显示需要转换为 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 设置Matplotlib显示样式
plt.figure(figsize=(12, 8))

# 显示原图
plt.subplot(1, 2, 1)  # 1行2列，第1个位置
plt.imshow(img_rgb)
plt.title("原图 (Original Image)", fontsize=12)
plt.axis("off")  # 关闭坐标轴

# ===================== 任务4：转换为灰度图并显示 =====================
# 彩色图转灰度图（cv2.COLOR_BGR2GRAY 自动处理通道转换）
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
plt.subplot(1, 2, 2)  # 1行2列，第2个位置
plt.imshow(gray_img, cmap="gray")  # cmap="gray" 指定灰度色板
plt.title("灰度图 (Grayscale Image)", fontsize=12)
plt.axis("off")

# 显示所有图像
plt.tight_layout()  # 调整子图间距
plt.show()

# ===================== 任务5：保存灰度图 =====================
gray_save_path = "gray_test.jpg"  # 灰度图保存路径
cv2.imwrite(gray_save_path, gray_img)
print(f"\n灰度图已保存至：{gray_save_path}")

# ===================== 任务6：NumPy 简单操作 =====================
print("\n===== NumPy 操作结果 =====")
# 操作1：输出指定像素值（示例：坐标 (100, 200)，注意OpenCV坐标是 (y, x) 即（行, 列））
pixel_y, pixel_x = 100, 200
original_pixel = img[pixel_y, pixel_x]  # 原图（BGR）该像素值
gray_pixel = gray_img[pixel_y, pixel_x]  # 灰度图该像素值
print(f"原图坐标 ({pixel_x}, {pixel_y}) 的像素值 (B, G, R)：{original_pixel}")
print(f"灰度图坐标 ({pixel_x}, {pixel_y}) 的像素值：{gray_pixel}")

# 操作2：裁剪左上角区域（示例：裁剪 200x200 像素的区域）
crop_size = 200
# NumPy 切片：img[行起始:行结束, 列起始:列结束]
crop_region = img[:crop_size, :crop_size]
# 保存裁剪后的区域
crop_save_path = "crop_top_left.jpg"
cv2.imwrite(crop_save_path, crop_region)
print(f"左上角 {crop_size}x{crop_size} 区域已保存至：{crop_save_path}")