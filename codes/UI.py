from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load data and models
df = pd.read_csv('..\data\cleaning_data\BatDongSan.csv')
gradient_boosted_trees = joblib.load('../models/gradient_boosted_trees_model.joblib')
random_forest = joblib.load('../models/random_forest_model.joblib')
decision_tree = joblib.load('../models/decision_tree_model.joblib')
glm = joblib.load('../models/Generalized_Linear_Model_model.pkl')
xgboost_model = joblib.load('../models/xgboost_model.joblib')

category_encoder = joblib.load('../models/category_encoder.pkl')
type_encoder = joblib.load('../models/type_encoder.pkl')
district_encoder = joblib.load('../models/district_encoder.pkl')
furnishing_encoder = joblib.load('../models/furnishing_encoder.pkl')

encoder_dict = {
    'category': category_encoder,
    'type': type_encoder,
    'Quận': district_encoder,
    'furnishing': furnishing_encoder
}


# Data preprocessing function
def preprocess_input_data(df, encoder_dict):
    categorical_cols = ['type', 'category', 'Quận', 'furnishing']
    encoded_data = []

    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
        encoder = encoder_dict[col]
        encoded_col = encoder.transform(df[col].values)
        encoded_data.append(encoded_col)

    numerical_cols = ['area', 'toilet', 'room', 'Số tầng']
    X_input = np.hstack([df[numerical_cols].values, np.array(encoded_data).T])
    return X_input


# Prediction function
def predict_price(model, X_input):
    price_pred = model.predict(X_input)
    return price_pred[0] if price_pred.ndim == 1 else price_pred


@app.route('/', methods=['GET', 'POST'])
def index():
    type_values = []  # Default empty list for type, will update after category selection
    gbt_price = rf_price = dt_price = glm_price = xgb_price = None

    if request.method == 'POST':
        # Get form data
        area = float(request.form['area'])
        toilet = int(request.form['toilet'])
        room = int(request.form['room'])
        floor = int(request.form['floor'])
        category = request.form['category']
        type_ = request.form['type']
        district = request.form['district']
        furnishing = request.form['furnishing']

        # Preprocess data
        df_input = pd.DataFrame({
            'area': [area],
            'toilet': [toilet],
            'room': [room],
            'Số tầng': [floor],
            'type': [type_],
            'category': [category],
            'Quận': [district],
            'furnishing': [furnishing]
        })

        X_input = preprocess_input_data(df_input, encoder_dict)

        # Predict prices using models
        gbt_price = predict_price(gradient_boosted_trees, X_input)
        rf_price = predict_price(random_forest, X_input)
        dt_price = predict_price(decision_tree, X_input)
        glm_price = predict_price(glm, X_input)
        xgb_price = predict_price(xgboost_model, X_input)

    # Get values for dropdowns
    category_values = df['category'].dropna().unique()
    district_values = df['Quận'].dropna().unique()
    furnishing_values = df['furnishing'].dropna().unique()

    return render_template('index.html',
                           category_values=category_values,
                           district_values=district_values,
                           furnishing_values=furnishing_values,
                           type_values=type_values,
                           gbt_price=gbt_price, rf_price=rf_price,
                           dt_price=dt_price, glm_price=glm_price,
                           xgb_price=xgb_price)


@app.route('/update_type', methods=['GET'])
def update_type():
    category = request.args.get('category')
    # Get corresponding types for the selected category
    type_values = df[df['category'] == category]['type'].dropna().unique().tolist()
    return jsonify(type_values)


if __name__ == '__main__':
    app.run(debug=True)
