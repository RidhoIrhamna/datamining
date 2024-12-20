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

# Dark Mode CSS
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


# Load your dataset here
df = pd.read_csv('Telco-Customer-Churn.csv')

# Sidebar untuk navigasi
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
        "KMeans Clustering"
    ],
    format_func=lambda x: f"📊 {x}"  # Menambahkan emoji pada menu
)

# Data Preprocessing
df.drop(columns='customerID', inplace=True)
df.isnull().sum()
df.dropna(inplace=True)

df['gender'] = df['gender'].replace({'F': 'Female', 'M': 'Male'})
df['gender'].nunique()

df['tenure'] = df['tenure'].astype(int)

# Convert Yes/No columns to 1/0
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
df[yes_no_columns] = df[yes_no_columns].replace({'Yes': 1, 'No': 0})

# Label encoding for categorical columns
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

# Feature Selection for Regression and Classification
X = df_num.drop(['Churn'], axis=1)
y = df_num['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Header utama
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

    # Membuat kolom-kolom untuk menampilkan grafik
    num_cols = 3  # Menampilkan 3 gambar per baris
    num_rows = (df.shape[1] // num_cols) + 1  # Menentukan jumlah baris

    columns = st.columns(num_cols)  # Membuat 3 kolom untuk layout

    col_idx = 0  # Index untuk menentukan kolom mana yang akan diisi dengan gambar

    for col in df.columns:
        if df[col].nunique() == 2:  # Jika kolom memiliki 2 nilai unik (misalnya kategorikal)
            value_counts = df[col].value_counts()

            # Membuat grafik pie dengan warna default
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Distribution of {col}')

            # Menampilkan grafik di kolom yang sesuai
            with columns[col_idx]:
                st.pyplot(fig)

            # Update index kolom
            col_idx += 1

            # Jika sudah selesai dengan 3 kolom, reset index dan pindah ke baris berikutnya
            if col_idx == num_cols:
                columns = st.columns(num_cols)  # Membuat kolom baru untuk baris berikutnya
                col_idx = 0  # Reset index kolom



elif menu == "Data Understanding Box Chart":
    st.subheader("Data Understanding")
    st.write("Menampilkan statistik deskriptif dan informasi dataset.")

    # Function to create barplot for columns with more than 2 unique values
    def create_barplot(data, col_name):
        fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size to ensure it's not too large
        sorted_counts = data.value_counts().sort_index()
        sorted_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel(f'{col_name}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col_name}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig

    # Get categorical columns with more than 2 unique values
    cols = df.select_dtypes(include=object).columns

    # Layout for displaying the plots
    num_cols = 3  # Number of columns per row
    col_idx = 0  # Index to keep track of columns in the row

    columns = st.columns(num_cols)  # Create columns for layout

    # For categorical columns with more than 2 unique values, create bar plots
    for col in cols:
        if len(df[col].unique()) > 2:  # Only columns with more than 2 unique values
            if len(df[col].unique()) < 5:  # Only consider columns with fewer than 5 unique values
                fig = create_barplot(df[col], col)
                
                # Display the plot in the current column
                with columns[col_idx]:
                    st.pyplot(fig)
                
                # Update column index
                col_idx += 1
                
                # Reset column index and create new row if we've filled 3 columns
                if col_idx == num_cols:
                    columns = st.columns(num_cols)  # Create new row of columns
                    col_idx = 0  # Reset column index for the new row

# Correlation Heatmap
elif menu == "Correlation Heatmap":
    st.subheader("Correlation HeatMap")
    # Menghitung korelasi antar kolom numerik
    correlation = df_num.corr()

    # Membuat figure untuk heatmap
    fig, ax = plt.subplots(figsize=(14, 12))  # Ukuran heatmap yang lebih kecil

    # Membuat heatmap dengan styling tambahan
    sns.heatmap(
        correlation, 
        annot=True,  # Menampilkan nilai pada setiap sel
        fmt='.2f',   # Format angka dua desimal
        cmap='coolwarm',  # Palet warna yang menarik
        linewidths=0.5,   # Menambahkan garis tipis antar sel
        linecolor='gray', # Warna garis antar sel
        cbar_kws={"shrink": 0.8},  # Mengatur ukuran color bar
        annot_kws={'size': 10, 'weight': 'bold', 'color': 'black'},  # Ukuran dan warna font angka (lebih kecil)
        square=True,  # Membuat heatmap menjadi persegi
        ax=ax         # Menyisipkan heatmap ke dalam subplot
    )

    # Menambahkan judul dan penataan tambahan pada plot
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', color='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10, color='white')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, color='white')

    # Menambahkan background gelap untuk seluruh figure
    fig.patch.set_facecolor('#2e3b4e')
    ax.set_facecolor('#2e3b4e')

    # Menampilkan heatmap dengan Streamlit
    st.pyplot(fig)

elif menu == "Churn Distribution by Feature":
    st.subheader("Churn Distribution by Feature")

    features = ['Contract', 'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity']

    # Create a 2-column layout for each row
    for i, feature in enumerate(features):
        # Group data by the selected feature and churn status
        churn_counts = df.groupby([feature, 'Churn']).size().unstack(fill_value=0)

        # Create a plot for each feature
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

        # Set the layout for light mode
        plt.tight_layout()
        fig = plt.gcf()
        fig.patch.set_facecolor('white')  # Light mode background
        ax.set_facecolor('white')  # Set axes background to white
        ax.tick_params(axis='both', which='major', labelsize=12, colors='black')  # Black ticks

        # Create a 2-column layout
        if i % 2 == 0:
            col1, col2 = st.columns(2)

        # Show the plot in the right column based on the index
        with [col1, col2][i % 2]:
            st.pyplot(fig)

elif menu == "Decision Tree Confusion Matrix":
    st.subheader("Decision Tree Confusion Matrix")

    # Model training and prediction
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    # Model Accuracy
    st.write("### Decision Tree Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred_tree)
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <h4 style='color: #4CAF50;'>Accuracy: {accuracy}</h4>
        </div>
    """, unsafe_allow_html=True)

    # Classification Report
    st.write("### Decision Tree Classification Report")
    report = classification_report(y_test, y_pred_tree, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    # Confusion Matrix
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

    # Ensure the model is trained
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    fig_tree = plt.figure(figsize=(50, 20))

    # Plotting the Decision Tree with default styling
    plot_tree(
        tree_model,
        filled=True,
        feature_names=X_train.columns,
        class_names=['No Churn', 'Churn']
    )

    # Set the title
    plt.title("Decision Tree Visualization", fontsize=24)

    # Show plot in Streamlit
    st.pyplot(fig_tree)

elif menu == "Feature Importance for Decision Tree Model":
    st.subheader("Feature Importance for Decision Tree Model")

    # Ensure the model is trained
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    # Calculate feature importances
    importances = tree_model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Display the DataFrame in Streamlit
    st.write("### Feature Importance Data")
    st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))

    # Visualize Feature Importance
    st.write("### Feature Importance for Decision Tree Model")
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title('Feature Importance for Decision Tree Model', fontsize=16, color='white')
    plt.xlabel('Importance', fontsize=12, color='white')
    plt.ylabel('Feature', fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    # Add gridlines
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Display the plot in Streamlit
    st.pyplot(plt)

# Logistic Regression Confusion Matrix
elif menu == "Logistic Regression Confusion Matrix":
    st.subheader("Logistic Regression Confusion Matrix")

    # Logistic Regression Model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    y_pred_logreg = logreg_model.predict(X_test)

    # Model Accuracy
    st.write("### Logistic Regression Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred_logreg)
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <h4 style='color: #4CAF50;'>Accuracy: {accuracy}</h4>
        </div>
    """, unsafe_allow_html=True)

    # Classification Report
    st.write("### Logistic Regression Classification Report")
    report = classification_report(y_test, y_pred_logreg, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'))

    # Confusion Matrix
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

    # Train Logistic Regression Model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)

    # Compute feature importances
    importances = logreg_model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(importances)
    }).sort_values(by='Importance', ascending=False)

    # Display the DataFrame in Streamlit
    st.write("### Feature Importance Data")
    st.dataframe(feature_importance_df.style.background_gradient(cmap='Blues'))

    # Create a dark mode style plot
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
    plt.title('Feature Importance for Logistic Regression Model', fontsize=16, color='white')
    plt.xlabel('Importance', fontsize=14, color='white')
    plt.ylabel('Feature', fontsize=14, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    # Render the plot
    st.pyplot(plt)


# Linear Regression: Actual vs Predicted (Monthly Charges)
elif menu == "Linear Regression: Actual vs Predicted":
    st.subheader("Linear Regression: Actual vs Predicted")
    # Data preparation
    X = df_num.drop(['MonthlyCharges'], axis=1)
    y = df_num['MonthlyCharges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.markdown("### Model Evaluation Metrics")
    st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px;'>
            <p style='color: #ffffff;'><b>Mean Squared Error (MSE):</b> <span style='color: #4CAF50;'>{mse:.2f}</span></p>
            <p style='color: #ffffff;'><b>Mean Absolute Error (MAE):</b> <span style='color: #4CAF50;'>{mae:.2f}</span></p>
            <p style='color: #ffffff;'><b>R-squared:</b> <span style='color: #4CAF50;'>{r2:.2f}</span></p>
        </div>
    """, unsafe_allow_html=True)

    # Plot Actual vs Predicted
    st.markdown("### Actual vs Predicted Plot")

    # Customize plot for light mode
    plt.style.use('default')  # Use default matplotlib style for light mode
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")  # Blue dots for data points
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Actual vs Predicted", fontsize=16, color='black')
    plt.xlabel("Actual Monthly Charges", fontsize=12, color='black')
    plt.ylabel("Predicted Monthly Charges", fontsize=12, color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')

    # Render plot in Streamlit
    st.pyplot(plt)

# Pair Plot
elif menu == "Pair Plot":
    st.subheader("Pair Plot")

    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['tenure', 'MonthlyCharges', 'TotalCharges'])

    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = df[['tenure', 'MonthlyCharges', 'TotalCharges']].astype(float)

    # Function to remove outliers using IQR
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

    # Pairplot with styling
    sns.set_theme(style="whitegrid")  # Set a clean and readable theme
    pairplot = sns.pairplot(
        df_num, 
        diag_kind='kde', 
        kind='scatter', 
        palette='viridis',  # Use a visually appealing color palette
        markers=["o", "s", "D"],  # Customize markers
        plot_kws={'alpha': 0.6, 's': 50}  # Adjust scatterplot transparency and size
    )

    # Customizing titles and labels
    pairplot.fig.suptitle('Pair Plot of Cleaned Data', 
                        y=1.02, fontsize=16, weight='bold', color='darkblue')
    pairplot.fig.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(pairplot.fig)

    # Print data shape after cleaning
    st.write(f"Data setelah pembersihan outlier: {df_num.shape}")

elif menu == "Elbow Method":
    st.subheader("Elbow Method for KMeans Clustering")
    
    # Path ke file gambar hasil Elbow Method
    elbow_image_path = "elbow_method.png"  # Ganti dengan path file yang benar

    try:
        from PIL import Image
        # Membuka dan menampilkan gambar
        elbow_image = Image.open(elbow_image_path)
        st.image(elbow_image, caption="Elbow Method for Optimal k", use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan. Pastikan file telah diekspor dari Jupyter Notebook.")

elif menu == "KMeans Clustering":
    st.subheader("KMeans Clustering Result")
    
    # Path ke file gambar hasil clustering
    image_path = "kmeans_clustering_results.png"  # Ganti dengan path file yang benar

    try:
        from PIL import Image
        # Membuka dan menampilkan gambar
        image = Image.open(image_path)
        st.image(image, caption="KMeans Clustering Results (Cleaned Data)", use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan. Pastikan file telah diekspor dari Jupyter Notebook.")
