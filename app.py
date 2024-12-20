import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

st.markdown("""
    <style>
    /* Styling global untuk tema gelap */
    .main {
        background-color: #1e1e1e;
        color: white;
    }
    .st-bk {
        background-color: #1e1e1e;
    }
    .stButton>button {
        background-color: #2d3748;
        color: white;
    }
    .css-1d391kg {
        color: white;
    }
    .css-1v3fvcr {
        color: white;
    }

    /* Styling tabel */
    .dataframe tbody tr:nth-child(odd) {
        background-color: #1e2a3a; /* Baris ganjil dengan background gelap */
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #2e3b4e; /* Baris genap dengan background sedikit lebih terang */
    }
    .dataframe th {
        background-color: #2d3748; /* Header tabel dengan background gelap */
        color: white; /* Teks putih pada header */
        font-weight: bold; /* Membuat teks header tebal */
        padding: 12px 8px; /* Padding di header */
        text-align: center; /* Teks header rata tengah */
    }
    .dataframe td {
        color: white; /* Warna teks putih di sel */
        padding: 10px 8px; /* Padding di sel */
        text-align: center; /* Rata tengah pada teks */
    }
    .dataframe {
        border: 1px solid #ccc; /* Border tabel */
        border-radius: 8px; /* Membulatkan sudut tabel */
        overflow: hidden; /* Memastikan sudut tabel tidak terpotong */
    }
    .dataframe tbody tr:hover {
        background-color: #4e5b6e; /* Efek hover dengan background sedikit terang */
    }
    .dataframe {
        width: 100%; /* Membuat tabel memenuhi lebar layar */
    }

    /* Styling untuk output info */
    .info-text {
        font-family: 'Courier New', monospace; /* Monospace font untuk output info */
        color: #f1f1f1; /* Teks putih agar lebih terlihat */
        background-color: #2e3b4e; /* Latar belakang gelap */
        padding: 10px;
        border-radius: 8px;
        font-size: 14px; /* Ukuran font lebih kecil */
        white-space: pre-wrap; /* Menjaga spasi dan format agar tetap terjaga */
        border: 1px solid #444; /* Border agar lebih terstruktur */
        overflow-x: auto; /* Scroll horizontal jika teks terlalu panjang */
    }

    /* Styling markdown */
    .stMarkdown {
        font-size: 14px;
        color: #f1f1f1;
        font-family: 'Arial', sans-serif;
    }

    /* Styling sidebar container */
    [data-testid="stSidebar"] {
        transition: all 0.3s ease-in-out;
        background-color: #2d3748; /* Background abu gelap untuk sidebar */
        color: white; /* Teks putih di sidebar */
        border-right: 1px solid #444;
    }
    
    /* Styling individual items in the sidebar */
    .sidebar-item {
        padding: 10px 15px;
        margin: 10px 0;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
        color: white; /* Teks putih */
        border-radius: 5px;
        background-color: #3a3f4b; /* Background abu gelap */
        transition: transform 0.2s ease, background-color 0.3s ease;
    }

    .sidebar-item:hover {
        transform: scale(1.1); /* Membesar saat hover */
        background-color: #4CAF50; /* Warna hijau saat hover */
        color: white; /* Tetap putih */
    }

    .sidebar-item a {
        text-decoration: none;
        color: inherit;
    }
    </style>
    """, unsafe_allow_html=True)


df = pd.read_csv('Telco-Customer-Churn.csv')

st.sidebar.title("Data Analysis App")
menu = st.sidebar.radio(
    "Pilih Analisis",
    [
        "Raw Data",
        "Info Data",
        "Data Understanding Pie Chart", 
        "Data Understanding Box Chart",
        "Correlation Heatmap", 
        "Churn Distribution by Feature",
        "Decision Tree Confusion Matrix", 
        "Decision Tree Visualization", 
        "Feature Importance for Decision Tree Model",
        "Logistic Regression Confusion Matrix", 
        "Feature Importance for Logistic Regression",
        "Linear Regression: Actual vs Predicted",
        "Pair Plot",
        "Elbow Method",
        "KMeans Clustering",
        "Prediksi Data"
    ],
    format_func=lambda x: f"📊 {x}"
)

df.drop(columns='customerID', inplace=True)
df.isnull().sum()
df.dropna(inplace=True)

df['gender'] = df['gender'].replace({'F': 'Female', 'M': 'Male'})
df['gender'].nunique()

df['tenure'] = df['tenure'].astype(int)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
df[yes_no_columns] = df[yes_no_columns].replace({'Yes': 1, 'No': 0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_num = df.copy()

for col in df.columns:
    if df[col].dtype == 'object':
        df_num[col] = le.fit_transform(df[col])

def find_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

outliers_df = pd.DataFrame()

for col in df_num:
    if len(df_num[col].unique()) > 2:
        outliers = find_outliers_iqr(df_num[col])
        if not outliers.empty:
            outliers_df = pd.concat([outliers_df, outliers.rename(col)], axis=1)

df_no_Outlier = df_num.copy()

def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

for col in outliers_df:
    df_no_Outlier[col] = remove_outliers(df_no_Outlier[col])

df_no_Outlier.dropna(inplace=True)

def find_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

numeric_cols = df_no_Outlier.select_dtypes(include=np.number).columns
outliers_df_2 = pd.DataFrame()

for col in numeric_cols:
    if len(df_no_Outlier[col].unique()) > 2:
        outliers = find_outliers_iqr(df_no_Outlier[col])
        if not outliers.empty:
            outliers_df_2 = pd.concat([outliers_df_2, outliers.rename(col)], axis=1)

df_num = df_no_Outlier

categorical_columns = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                    'StreamingMovies', 'Contract','PaymentMethod']
df_one_hot = pd.get_dummies(df_num, columns=categorical_columns, drop_first=True)

X = df_num.drop(['Churn'], axis=1)
y = df_num['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

st.title("Data Science Analysis")
st.write("Aplikasi ini membantu Anda menganalisis data dengan berbagai metode analisis.")

if menu == "Raw Data":
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")
    st.dataframe(df)

elif menu == "Info Data":
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.markdown(f'<div class="info-text">{info_str}</div>', unsafe_allow_html=True)

elif menu == "Data Understanding Pie Chart":
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")

    num_cols = 3
    num_rows = (df.shape[1] // num_cols) + 1

    columns = st.columns(num_cols)

    col_idx = 0

    for col in df.columns:
        if df[col].nunique() == 2:
            value_counts = df[col].value_counts()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Distribution of {col}')

            with columns[col_idx]:
                st.pyplot(fig)

            col_idx += 1

            if col_idx == num_cols:
                columns = st.columns(num_cols)
                col_idx = 0



elif menu == "Data Understanding Box Chart":
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")

    def create_barplot(data, col_name):
        fig, ax = plt.subplots(figsize=(5, 5))
        sorted_counts = data.value_counts().sort_index()
        sorted_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(f'{col_name}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col_name}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig

    cols = df.select_dtypes(include=object).columns

    num_cols = 3
    col_idx = 0

    columns = st.columns(num_cols)

    for col in cols:
        if len(df[col].unique()) > 2:
            if len(df[col].unique()) < 5:
                fig = create_barplot(df[col], col)

                with columns[col_idx]:
                    st.pyplot(fig)

                col_idx += 1

                if col_idx == num_cols:
                    columns = st.columns(num_cols)
                    col_idx = 0

elif menu == "Correlation Heatmap":
    st.subheader("Correlation HeatMap")
    correlation = df_num.corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        correlation, 
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.8},
        annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'},
        square=True,
        ax=ax
    )

    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10, color='white')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, color='white')

    fig.patch.set_facecolor('#2e3b4e')
    ax.set_facecolor('#2e3b4e')

    st.pyplot(fig)

elif menu == "Churn Distribution by Feature":
    st.subheader("Churn Distribution by Feature")

    features = ['Contract', 'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity']

    for i, feature in enumerate(features):
        churn_counts = df.groupby([feature, 'Churn']).size().unstack(fill_value=0)

        ax = churn_counts.plot(
            kind='bar',
            stacked=True,
            color=['#0d3a59', '#1f77b4'],
            figsize=(8, 6),
        )

        ax.set_title(f'Churn Distribution by {feature}', fontsize=14, color='black', weight='bold')
        ax.set_ylabel("Number of Customers", fontsize=12, color='black')
        ax.set_xlabel(feature, fontsize=12, color='black')
        ax.legend(title='Churn', labels=['No', 'Yes'], fontsize=10, title_fontsize=12)

        plt.tight_layout()
        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=12, colors='black')

        if i % 2 == 0:
            col1, col2 = st.columns(2)

        with [col1, col2][i % 2]:
            st.pyplot(fig)

elif menu == "Decision Tree Confusion Matrix":
    st.subheader("Decision Tree Confusion Matrix")

    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    st.write("### Decision Tree Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred_tree)
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <h4 style='color: #4CAF50;'>Accuracy: {accuracy}</h4>
        </div>
    """, unsafe_allow_html=True)

    st.write("### Decision Tree Classification Report")
    report = classification_report(y_test, y_pred_tree, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    cm_tree = confusion_matrix(y_test, y_pred_tree)

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], cbar=False)
    plt.title("Decision Tree Confusion Matrix", fontsize=16, color='white')
    plt.xlabel("Predicted", fontsize=12, color='white')
    plt.ylabel("Actual", fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    st.pyplot(plt)

elif menu == "Decision Tree Visualization":
    st.subheader('Decision Tree Visualization')

    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    fig_tree = plt.figure(figsize=(50, 20))

    plot_tree(
        tree_model,
        filled=True,
        feature_names=X_train.columns,
        class_names=['No Churn', 'Churn']
    )

    plt.title("Decision Tree Visualization", fontsize=24)

    st.pyplot(fig_tree)

elif menu == "Feature Importance for Decision Tree Model":
    st.subheader("Feature Importance for Decision Tree Model")

    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    importances = tree_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance Data")
    st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))

    st.write("### Feature Importance for Decision Tree Model")
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title('Feature Importance for Decision Tree Model', fontsize=16, color='white')
    plt.xlabel('Importance', fontsize=12, color='white')
    plt.ylabel('Feature', fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    st.pyplot(plt)

elif menu == "Logistic Regression Confusion Matrix":
    st.subheader("Logistic Regression Confusion Matrix")

    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    y_pred_logreg = logreg_model.predict(X_test)

    st.write("### Logistic Regression Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred_logreg)
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <h4 style='color: #4CAF50;'>Accuracy: {accuracy}</h4>
        </div>
    """, unsafe_allow_html=True)

    st.write("### Logistic Regression Classification Report")
    report = classification_report(y_test, y_pred_logreg, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    cm_logreg = confusion_matrix(y_test, y_pred_logreg)

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], cbar=False)
    plt.title("Logistic Regression Confusion Matrix", fontsize=16, color='white')
    plt.xlabel("Predicted", fontsize=12, color='white')
    plt.ylabel("Actual", fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    st.pyplot(plt)

elif menu == "Feature Importance for Logistic Regression":
    st.subheader("Feature Importance for Logistic Regression")

    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    importances = logreg_model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(importances)
    }).sort_values(by='Importance', ascending=False)

    st.write("### Feature Importance Data")
    st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title('Feature Importance for Logistic Regression Model', fontsize=16, color='white')
    plt.xlabel('Importance', fontsize=14, color='white')
    plt.ylabel('Feature', fontsize=14, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    st.pyplot(plt)

elif menu == "Linear Regression: Actual vs Predicted":
    st.subheader("Linear Regression: Actual vs Predicted")

    X = df_num.drop(['MonthlyCharges'], axis=1)
    y = df_num['MonthlyCharges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### Model Evaluation Metrics")
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <p style='color: #ffffff;'><b>Mean Squared Error (MSE):</b> <span style='color: #4CAF50;'>{mse:.2f}</span></p>
            <p style='color: #ffffff;'><b>Mean Absolute Error (MAE):</b> <span style='color: #4CAF50;'>{mae:.2f}</span></p>
            <p style='color: #ffffff;'><b>R-squared:</b> <span style='color: #4CAF50;'>{r2:.2f}</span></p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Actual vs Predicted Plot")

    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted", fontsize=16, color='black')
    plt.xlabel("Actual Monthly Charges", fontsize=12, color='black')
    plt.ylabel("Predicted Monthly Charges", fontsize=12, color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')

    st.pyplot(plt)

elif menu == "Pair Plot":
    st.subheader("Pair Plot")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['tenure', 'MonthlyCharges', 'TotalCharges'])

    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']].astype(float)

    def remove_outliers_iqr(data, columns):
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        return data

    
    columns_to_clean = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_clean = remove_outliers_iqr(df, columns_to_clean)

    df_num = df_clean[columns_to_clean]

    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(
        df_num, 
        diag_kind='kde', 
        kind='scatter', 
        palette='viridis',
        markers=["o", "s", "D"],
        plot_kws={'alpha': 0.6, 's': 50}
    )

    pairplot.fig.suptitle('Pair Plot of Cleaned Data', 
                        y=1.02, fontsize=16, weight='bold', color='darkblue')
    pairplot.fig.tight_layout()

    st.pyplot(pairplot.fig)

    st.write(f"Data setelah pembersihan outlier: {df_num.shape}")

elif menu == "Elbow Method":
    st.subheader("Elbow Method for KMeans Clustering")

    elbow_image_path = "elbow_method.png" 

    try:
        from PIL import Image
        elbow_image = Image.open(elbow_image_path)
        st.image(elbow_image, caption="Elbow Method for Optimal k", use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan. Pastikan file telah diekspor dari Jupyter Notebook.")

elif menu == "KMeans Clustering":
    st.subheader("KMeans Clustering Result")

    image_path = "kmeans_clustering_results.png"

    try:
        from PIL import Image
        image = Image.open(image_path)
        st.image(image, caption="KMeans Clustering Results (Cleaned Data)", use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan. Pastikan file telah diekspor dari Jupyter Notebook.")

elif menu == "Prediksi Data":
    st.subheader("Prediksi Data Lebih Lanjut")

    model = joblib.load('logistic_model.pkl')

    st.markdown("""
    Aplikasi ini memprediksi apakah seorang pelanggan akan churn berdasarkan beberapa fitur penting:
    - **Tenure**: Berapa lama pelanggan menggunakan layanan.
    - **Contract**: Jenis kontrak pelanggan.
    - **MonthlyCharges**: Biaya bulanan pelanggan.
    - **PaperlessBilling**: Apakah pelanggan menggunakan tagihan tanpa kertas.
    - **PaymentMethod**: Metode pembayaran pelanggan.
    """)

    tenure = st.number_input("Tenure (bulan):", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract:", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges:", min_value=0.0, max_value=1000.0, value=50.0)
    paperless_billing = st.selectbox("Paperless Billing:", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    if st.button("Predict Churn"):
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'Contract': [contract],
            'MonthlyCharges': [monthly_charges],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method]
        })
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        if prediction[0] == 1:
            st.error(f"Pelanggan diprediksi akan churn dengan probabilitas {prediction_proba[0][1]:.2f}.")
        else:
            st.success(f"Pelanggan diprediksi tidak akan churn dengan probabilitas {prediction_proba[0][0]:.2f}.")