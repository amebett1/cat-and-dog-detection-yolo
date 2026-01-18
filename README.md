# 🐶 🐱 Dog vs. Cat Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-purple)

## 📖 Giới thiệu (Overview)
Dự án này là một mô hình Deep Learning được xây dựng để phân loại (hoặc phát hiện) chó và mèo từ hình ảnh hoặc video. Hệ thống sử dụng YOLO để trích xuất đặc trưng và đưa ra dự đoán với độ chính xác cao.

**Mục tiêu chính:**
- Phân biệt chính xác giữa chó và mèo trong các điều kiện ánh sáng và góc chụp khác nhau.
- Tối ưu hóa thời gian huấn luyện (Training) sử dụng NVIDIA GPU (CUDA).

## Tính năng (Features)
- 🖼️ **Image/Video Classification:** Dự đoán nhãn (Chó/Mèo) cho từng ảnh/video đầu vào.
- 📊 **Visualized Metrics:** Biểu đồ Loss/Accuracy trực quan trong quá trình train.

## 🛠️ Cài đặt (Installation)

### 1. Yêu cầu hệ thống
- Python 3.8+
- NVIDIA Driver (nếu dùng GPU)
- CUDA Toolkit (được cài tự động qua PyTorch/TensorFlow)

### 2. 
