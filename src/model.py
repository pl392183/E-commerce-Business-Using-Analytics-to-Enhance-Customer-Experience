import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv('file_temp.csv')

def scaler_data(df):
    le = {}
    for column in df.columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        le[column] = label_encoder
    return le

def predict_customer_return(new_df,customer_id, transaction_date):
    new_df['Previous_Transaction_Date'] = new_df.groupby('CustomerID')['Transaction_Date'].shift(1)
    new_df['Days_Between_Purchases'] = (pd.to_datetime(new_df['Transaction_Date']) - pd.to_datetime(new_df['Previous_Transaction_Date'])).dt.days

    new_df = new_df[new_df['Days_Between_Purchases'].notnull()]
    features = ['Tenure_Months', 'Days_Between_Purchases', 'Avg_Price']
    X = new_df[features]
    y = new_df['Days_Between_Purchases'].shift(-1)

    X = X[:-1]
    y = y[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    customer_history = new_df[new_df['CustomerID'] == int(customer_id)]
    if not customer_history.empty:
        last_transaction = customer_history.iloc[-1]
        tenure_months = last_transaction['Tenure_Months']
        avg_price = customer_history['Avg_Price'].mean()
        last_transaction_date = last_transaction['Transaction_Date']
        days_between_purchases = (pd.to_datetime(transaction_date) - pd.to_datetime(last_transaction_date)).days

        input_features = pd.DataFrame([{
            'Tenure_Months': tenure_months,
            'Days_Between_Purchases': days_between_purchases,
            'Avg_Price': avg_price
        }])
        predict_return = model.predict(input_features)
        return f"Dự đoán khách hàng {customer_id} sẽ quay lại sau khoảng {predict_return[0]:.2f} ngày."
    else:
        return f"Không tìm thấy lịch sử giao dịch cho khách hàng {customer_id}."

def predict_product_for_customer(df, gender, age, category, city, state, month):
    new_df = df.copy()
    le = scaler_data(new_df)
    new_df = new_df.drop(['CustomerID', 'Delivery_Charges','Coupon_Code','Discount_pct','GST','Coupon_Status'],axis=1)
    df_temp = new_df[['Gender','Customer_Age', 'Product_Category', 'Product_Description', 'City', 'State', 'Month']]
    X = df_temp.drop('Product_Description', axis=1)
    y = df_temp['Product_Description']
    scaler = MinMaxScaler()
    model = RandomForestClassifier(random_state=42)
    X_train = scaler.fit_transform(X)
    model.fit(X_train, y)
    data = pd.DataFrame({
        'Gender': [gender],
        'Customer_Age': [age],
        'Product_Category': [category],
        'City': [city],
        'State': [state],
        'Month': [month]
    })
    data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
    data['State'] = LabelEncoder().fit_transform(data['State'])
    data['Month'] = LabelEncoder().fit_transform(data['Month'])
    data['City'] = LabelEncoder().fit_transform(data['City'])
    data['Product_Category'] = LabelEncoder().fit_transform(data['Product_Category'])
    data = data[['Gender', 'Customer_Age', 'Product_Category', 'City', 'State', 'Month']]
    data_scaled = scaler.transform(data)
    predict_product = model.predict(data_scaled)
    product = le['Product_Description'].inverse_transform(predict_product)[0]
    return f'Predicted Product: {product}'

def predict_potential_customer(df, gender, age, state, tenure, spend):
    new_df = df.copy()
    le = scaler_data(new_df)
    new_df = new_df[['Gender', 'Customer_Age', 'State', 'Tenure_Months', 'Total_Spend']]
    new_df['Gender'] = LabelEncoder().fit_transform(new_df['Gender'])
    new_df['State'] = LabelEncoder().fit_transform(new_df['State'])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(new_df)
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Segment'] = kmeans.fit_predict(X_scaled)

    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[kmeans.labels_], axis=1)

    df['Distance'] = distances
    threshold = df['Distance'].mean() + df['Distance'].std()
    df['Potential_Customer'] = np.where(df['Distance'] < threshold, "Potential", "Non-Potential")

    X_po = new_df[['Gender', 'Customer_Age', 'State', 'Tenure_Months', 'Total_Spend']]
    y_po = df['Potential_Customer']

    le['Potential_Customer'] = LabelEncoder()
    y_po = le['Potential_Customer'].fit_transform(y_po)
    print(X_po)
    scaler = MinMaxScaler()
    X_po = scaler.fit_transform(X_po)

    model_po = RandomForestClassifier(random_state=42)
    model_po.fit(X_po, y_po)
    # Tạo dataframe mới từ dữ liệu đầu vào
    data = pd.DataFrame({
        'Gender': [gender],
        'Customer_Age': [age],
        'State': [state],
        'Tenure_Months': [tenure],
        'Total_Spend': [spend]
    })
    data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
    data['State'] = LabelEncoder().fit_transform(data['State'])
    new_data_scaled = scaler.transform(data)
    predict = model_po.predict(new_data_scaled)
    prd = le['Potential_Customer'].inverse_transform(predict)[0]
    return f"The customer is a {prd} Customer for the bussiness"
