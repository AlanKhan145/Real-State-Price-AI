import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
# Đọc dữ liệu từ file CSV
df = pd.read_csv('../data/cleaning_data/BatDongSan.csv')

# Hàm nhập dữ liệu đầu vào từ người dùng dựa trên DataFrame
def get_input_data_from_df(df):
    print("Nhập các thuộc tính bất động sản:")

    # Lấy các giá trị duy nhất từ các cột phân loại
    category_values = df['category'].dropna().unique()
    district_values = df['Quận'].dropna().unique()
    furnishing_values = df['furnishing'].dropna().unique()

    # Gợi ý và chọn giá trị cho loại hình
    print("\nCác giá trị hợp lệ cho Loại hình:")
    for idx, value in enumerate(category_values, 1):
        print(f"{idx}. {value}")
    category_idx = int(input("Chọn Loại hình (Nhập số thứ tự): ")) - 1
    category_ = category_values[category_idx]

    # Lọc giá trị 'type' theo loại hình đã chọn
    type_values = df[df['category'] == category_]['type'].dropna().unique()
    print("\nCác giá trị hợp lệ cho Danh mục:")
    for idx, value in enumerate(type_values, 1):
        print(f"{idx}. {value}")
    type_idx = int(input("Chọn Danh mục (Nhập số thứ tự): ")) - 1
    property_type = type_values[type_idx]

    # Chọn quận
    print("\nCác giá trị hợp lệ cho Quận:")
    for idx, value in enumerate(district_values, 1):
        print(f"{idx}. {value}")
    district_idx = int(input("Chọn Quận (Nhập số thứ tự): ")) - 1
    district = district_values[district_idx]

    # Chọn tình trạng nội thất
    print("\nCác giá trị hợp lệ cho Tình trạng nội thất:")
    for idx, value in enumerate(furnishing_values, 1):
        print(f"{idx}. {value}")
    furnishing_idx = int(input("Chọn Tình trạng nội thất (Nhập số thứ tự): ")) - 1
    furnishing = furnishing_values[furnishing_idx]

    # Nhập các thông số khác
    area = float(input("Diện tích (m2): "))
    toilet = int(input("Số toilet: "))
    room = int(input("Số phòng ngủ: "))
    floor = int(input("Số tầng: "))

    return pd.DataFrame({
        'area': [area],
        'toilet': [toilet],
        'room': [room],
        'Số tầng': [floor],
        'type': [property_type],
        'category': [category_],
        'Quận': [district],
        'furnishing': [furnishing]
    })


# Hàm tiền xử lý dữ liệu đầu vào sử dụng LabelEncoder
def preprocess_input_data(df, encoder_dict):
    categorical_cols = ['type', 'category', 'Quận', 'furnishing']
    encoded_data = []

    for col in categorical_cols:
        encoder = encoder_dict[col]
        encoded_col = encoder.transform(df[col].values)
        encoded_data.append(encoded_col)

    # Gộp dữ liệu mã hóa với các cột số
    numerical_cols = ['area', 'toilet', 'room', 'Số tầng']
    X_input = np.hstack([df[numerical_cols].values, np.array(encoded_data).T])
    return X_input


# Hàm dự đoán giá bất động sản
def predict_price(model, X_input):
    price_pred = model.predict(X_input)
    return price_pred[0] if price_pred.ndim == 1 else price_pred


# Tải các mô hình đã huấn luyện
gradient_boosted_trees = joblib.load('../models/gradient_boosted_trees_model.joblib')
random_forest = joblib.load('../models/random_forest_model.joblib')
decision_tree = joblib.load('../models/decision_tree_model.joblib')
glm = joblib.load('../models/Generalized_Linear_Model_model.pkl')
xgboost_model = joblib.load('../models/xgboost_model.joblib')

# Tải các encoder đã huấn luyện
category_encoder = joblib.load('../models/category_encoder.pkl')
type_encoder = joblib.load('../models/type_encoder.pkl')
district_encoder = joblib.load('../models/district_encoder.pkl')
furnishing_encoder = joblib.load('../models/furnishing_encoder.pkl')

# Lưu các encoder vào một dictionary
encoder_dict = {
    'category': category_encoder,
    'type': type_encoder,
    'Quận': district_encoder,
    'furnishing': furnishing_encoder
}

# Lấy dữ liệu đầu vào từ người dùng
df_input = get_input_data_from_df(df)

# Tiền xử lý dữ liệu đầu vào
X_input = preprocess_input_data(df_input, encoder_dict)

# Dự đoán giá bất động sản bằng các mô hình
gbt_price = predict_price(gradient_boosted_trees, X_input)
rf_price = predict_price(random_forest, X_input)
dt_price = predict_price(decision_tree, X_input)
glm_price = predict_price(glm, X_input)
xgb_price = predict_price(xgboost_model, X_input)

# Hiển thị kết quả
print("\nDự đoán giá bất động sản:")
print(f"1. Gradient Boosted Trees: {gbt_price:.2f} tỷ VND")
print(f"2. Random Forest: {rf_price:.2f} tỷ VND")
print(f"3. Decision Tree: {dt_price:.2f} tỷ VND")
print(f"4. Generalized Linear Model: {glm_price:.2f} tỷ VND")
print(f"5. XGBoost Model: {xgb_price:.2f} tỷ VND")
