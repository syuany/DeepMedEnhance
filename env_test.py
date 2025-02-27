import pydicom
import cv2
import torch
import SimpleITK as sitk

print("[✓] Python版本验证:")
print(f"Python版本: {torch.__version__}")  # 应显示3.8/3.9

print("\n[✓] DICOM处理验证:")
dcm = pydicom.dcmread("test.dcm")  # 替换为你的DICOM文件路径
print(f"DICOM信息: {dcm.PatientName} (Modality: {dcm.Modality})")

print("\n[✓] OpenCV验证:")
img = cv2.cvtColor(cv2.imread("test.png"), cv2.COLOR_BGR2GRAY)  # 测试图像处理
print(f"OpenCV版本: {cv2.__version__}, 图像尺寸: {img.shape}")

print("\n[✓] PyTorch验证:")
print(f"PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
x = torch.randn(3,3)
print(f"张量运算测试: {x @ x.t()}")

print("\n[✓] SimpleITK验证:")
print(f"SimpleITK版本: {sitk.Version_VersionString()}")
