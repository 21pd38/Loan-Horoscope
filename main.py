from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Load the loan_approval.csv file
df = pd.read_csv('loan_approval.csv')

# Split the df into input features (X) and target variable (y)
X = df.drop(columns=['loan_status', 'loan_id'])
y = df['loan_status']

# Data Preprocessing: Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the df into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store accuracies for each kernel
accuracies = {}

kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    # Load the SVM model from the pickle file
    model_filename = f'svm_model_{kernel}.pkl'
    with open(model_filename, 'rb') as file:
        svm_model = pickle.load(file)

    # Use the SVM model to make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[kernel] = accuracy

@app.route('/')
def home():
    return render_template('home.html', accuracies=accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    user_input = [
        float(request.form['no_of_dependents']),
        float(request.form['education']),
        float(request.form['self_employed']),
        float(request.form['income_annum']),
        float(request.form['loan_amount']),
        float(request.form['loan_term']),
        float(request.form['cibil_score']),
        float(request.form['residential_assets_value']),
        float(request.form['commercial_assets_value']),
        float(request.form['luxury_assets_value']),
        float(request.form['bank_asset_value'])
    ]

    # Use the SVM model to make a prediction based on user input
    predictions = {}
    for kernel in kernels:
        with open(f'svm_model_{kernel}.pkl', 'rb') as file:
            svm_model = pickle.load(file)
        prediction = svm_model.predict([user_input])
        predictions[kernel] = "Approved" if prediction[0] == 0 else "Rejected"

    return render_template('predict.html', predictions=predictions)

def get_image():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/visualizations')
def visualizations():
    # Create a figure and a list to store the image paths
    fig = plt.figure(figsize=(14, 14))
    image_paths = []

    # Visualization 1: Bar Chart - Loan Approval Rate by Education
    ax1 = plt.subplot(5, 2, 1)
    bar_data1 = df.groupby('education')['loan_status'].mean().reset_index()
    sns.barplot(data=bar_data1, x='education', y='loan_status', palette='viridis', ax=ax1)
    ax1.set_title('Loan Approval Rate by Education')
    #image_paths.append('static/bar_chart.png')

    # Visualization 2: Pie Chart - Education Distribution
    ax2 = plt.subplot(5, 2, 2)
    education_count = df['education'].value_counts().reset_index()
    education_count.columns = ['Education', 'Count']
    ax2.pie(education_count['Count'], labels=education_count['Education'], autopct='%1.1f%%', shadow=True, colors=['#ff9999', '#66b3ff'])
    ax2.set_title('Education Distribution')
    #image_paths.append('static/pie_chart.png')

    # Visualization 3: Scatter Plot - Income vs. Loan Amount
    ax3 = plt.subplot(5, 2, 3)
    sns.scatterplot(data=df, x='income_annum', y='loan_amount', hue='loan_status', palette='coolwarm', s=50, ax=ax3)
    ax3.set_title('Income vs. Loan Amount')
    #image_paths.append('static/scatter_plot.png')

    # Visualization 4: Box Plot - Loan Amount by Education
    ax4 = plt.subplot(5, 2, 4)
    sns.boxplot(data=df, x='education', y='loan_amount', palette='Set2', ax=ax4)
    ax4.set_title('Loan Amount by Education')
    #image_paths.append('static/box_plot.png')

    # Visualization 5: Confusion Matrix
    ax5 = plt.subplot(5, 2, 5)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d', ax=ax5)
    ax5.set_title('Confusion Matrix')
    #image_paths.append('static/confusion_matrix.png')

    # Visualization 6: Bar Chart - Loan Approval Rate by Employment Status
    ax6 = plt.subplot(5, 2, 6)
    bar_data2 = df.groupby('self_employed')['loan_status'].mean().reset_index()
    sns.barplot(data=bar_data2, x='self_employed', y='loan_status', palette='mako', ax=ax6)
    ax6.set_title('Loan Approval Rate by Employment Status')
    #image_paths.append('static/bar_chart2.png')

    # Visualization 7: Scatter Plot - CIBIL Score vs. Bank Asset Value
    ax7 = plt.subplot(5, 2, 7)
    sns.scatterplot(data=df, x='cibil_score', y='bank_asset_value', hue='loan_status', palette='Set3', s=50, ax=ax7)
    ax7.set_title('CIBIL Score vs. Bank Asset Value')
    #image_paths.append('static/scatter_plot3.png')

    # Visualization 8: Box Plot - Residential Assets by Loan Status
    ax8 = plt.subplot(5, 2, 8)
    sns.boxplot(data=df, x='loan_status', y='residential_assets_value', palette='Set1', ax=ax8)
    ax8.set_title('Residential Assets by Loan Status')
    #image_paths.append('static/box_plot2.png')

    # Visualization 9: Pie Chart - Employment Status Distribution
    ax9 = plt.subplot(5, 2, 9)
    employment_count = df['self_employed'].value_counts().reset_index()
    employment_count.columns = ['Employment Status', 'Count']
    ax9.pie(employment_count['Count'], labels=employment_count['Employment Status'], autopct='%1.1f%%', shadow=True, colors=['#ff9999', '#66b3ff'])
    ax9.set_title('Employment Status Distribution')
    #image_paths.append('static/pie_chart2.png')

    # Visualization 10: Scatter Plot - Commercial Assets vs. Residential Assets
    ax10 = plt.subplot(5, 2, 10)
    sns.scatterplot(data=df, x='commercial_assets_value', y='residential_assets_value', hue='loan_status', palette='muted', s=50, ax=ax10)
    ax10.set_title('Commercial vs. Residential Assets')
    #image_paths.append('static/scatter_plot4.png')

    # Save the figure
    #fig.tight_layout()

    for i, image_path in enumerate(image_paths):
        #plt.savefig(image_path)
        x = 1

    return render_template('visualizations.html')

if __name__ == '__main__':
    app.run(debug=True)
