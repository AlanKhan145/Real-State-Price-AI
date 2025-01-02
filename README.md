# Real-State-Price-AI
 Dự án này sử dụng các mô hình học máy để dự đoán giá bất động sản dựa trên các đặc điểm đầu vào như diện tích, vị trí, số tầng, số phòng và các yếu tố khác ảnh hưởng đến giá trị của bất động sản. Mục tiêu là phát triển một mô hình có khả năng dự đoán chính xác giá bất động sản, giúp người dùng đưa ra quyết định thông minh hơn khi mua bán hoặc đầu tư vào bất động sản.

Hướng dẫn cài đặt

1. Quá trình tiền xử lý dữ liệu:

Tải file codes/RealEstate_Data_Cleaning.ipynb lên môi trường Google Colab cá nhân.

Tải bộ dữ liệu gốc từ thư mục data/raw_data.

Thực hiện làm sạch dữ liệu trên Google Colab.

Lưu file dữ liệu đã tiền xử lý vào thư mục data/cleaning_data/BatDongSan.csv.

2. Quá trình huấn luyện mô hình:

Tải file codes/Real_Estate_Modeling.ipynb lên môi trường Google Colab cá nhân.

Tải bộ dữ liệu đã tiền xử lý từ thư mục data/raw_data để làm dữ liệu huấn luyện.

Huấn luyện mô hình trên Google Colab.

Lưu các mô hình đã huấn luyện vào thư mục models.

3. Khởi chạy giao diện:

Trên máy tính cá nhân, cài đặt các thư viện cần thiết bằng cách chạy:

pip install -r requirements.txt

Chạy chương trình giao diện người dùng:

cd codes

python UI.py

Giao diện web sẽ được chạy tại: http://127.0.0.1:5000
