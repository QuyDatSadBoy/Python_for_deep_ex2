# Python for Deep Learning - Exercise 2

Đây là bài tập thực hành số 2 về Deep Learning sử dụng PyTorch, tập trung vào việc xây dựng và triển khai một mạng neural network đa lớp (Multi-layer Neural Network).

## 📋 Mô tả dự án

Bài tập này yêu cầu sinh viên triển khai:
1. **Hàm kích hoạt (Activation Functions)**: Sigmoid, Tanh, ReLU, hoặc Leaky ReLU
2. **Hàm Softmax**: Để chuyển đổi output thành phân phối xác suất
3. **Forward Pass**: Tính toán lan truyền thuận qua toàn bộ mạng neural

## 🏗️ Kiến trúc mạng Neural Network

Mạng neural được thiết kế với kiến trúc sau:

```
Input Layer: 784 nodes (28x28 pixels - tương ứng với ảnh MNIST)
    ↓
Hidden Layer 1: 128 nodes
    ↓
Hidden Layer 2: 256 nodes  
    ↓
Hidden Layer 3: 128 nodes
    ↓
Output Layer: 10 nodes (10 classes phân loại)
```

## 📁 Cấu trúc thư mục

```
Python_for_deep_ex2/
├── deeplearning_ex2.py          # File bài tập (cần hoàn thành)
├── deeplearning_ex2_solution.py # File đáp án tham khảo
└── README.md                    # File này
```

## 🔧 Yêu cầu hệ thống

- Python 3.7+
- PyTorch 1.0+

### Cài đặt dependencies

```bash
pip install torch
```

## 📝 Chi tiết bài tập

### 1. Triển khai hàm kích hoạt (activation_func)

Cần triển khai một trong các hàm kích hoạt sau:

- **Sigmoid**: `σ(x) = 1/(1 + e^(-x))`
- **Tanh**: `tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))`
- **ReLU**: `ReLU(x) = max(0, x)`
- **Leaky ReLU**: `LeakyReLU(x) = max(εx, x)` với ε = 0.01

```python
def activation_func(x):
    #TODO Implement one of these following activation function: sigmoid, tanh, ReLU, leaky ReLU
    epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    result = None
    return result
```

### 2. Triển khai hàm Softmax

Hàm Softmax chuyển đổi vector đầu ra thành phân phối xác suất:

```python
def softmax(x):
    # TODO Implement softmax function here
    result = None
    return result
```

**Công thức Softmax**: `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`

### 3. Tính toán Forward Pass

Cần triển khai quá trình lan truyền thuận qua toàn bộ mạng:

```python
#TODO Calculate forward pass of the network here. Result should have the shape of [1,10]
# Dont forget to check if sum of result = 1.0
result = None
```

**Quy trình Forward Pass**:
1. `h1 = activation_func(input_data × W1 + B1)`
2. `h2 = activation_func(h1 × W2 + B2)`
3. `h3 = activation_func(h2 × W3 + B3)`
4. `output = softmax(h3 × W4 + B4)`

## 🎯 Mục tiêu học tập

Sau khi hoàn thành bài tập, sinh viên sẽ:

- ✅ Hiểu cách hoạt động của các hàm kích hoạt khác nhau
- ✅ Nắm vững cách triển khai hàm Softmax
- ✅ Biết cách tính toán Forward Pass trong mạng neural đa lớp
- ✅ Làm quen với thư viện PyTorch cho Deep Learning
- ✅ Hiểu kiến trúc của mạng neural network cơ bản

## 🚀 Cách chạy code

### Chạy file bài tập (sau khi hoàn thành):
```bash
python deeplearning_ex2.py
```

### Chạy file đáp án để tham khảo:
```bash
python deeplearning_ex2_solution.py
```

## ✅ Kiểm tra kết quả

Kết quả đúng phải thỏa mãn:
- Shape của output: `[1, 10]`
- Tổng các phần tử trong output = 1.0 (do sử dụng Softmax)
- Tất cả các giá trị đều nằm trong khoảng [0, 1]

## 💡 Gợi ý triển khai

### Hàm Sigmoid (đề xuất):
```python
def activation_func(x):
    return 1 / (1 + torch.exp(-x))
```

### Hàm Softmax:
```python
def softmax(x):
    max_value, _ = torch.max(x, dim=1)
    norm = torch.exp(x - max_value)
    return norm / norm.sum()
```

## 📚 Tài liệu
