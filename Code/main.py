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

    # Visualization 1: Bar Chart - No. of dependants
    dependents_counts = df['no_of_dependents'].value_counts().sort_index()

    # Create a bar chart for the number of dependents
    plt.figure(figsize=(8, 6))
    plt.bar(dependents_counts.index, dependents_counts, color='skyblue')
    plt.xlabel('Number of Dependents')
    plt.ylabel('Count')
    plt.title('Distribution of Number of Dependents')
    plt.xticks(dependents_counts.index)
    #plt.show()

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

    # Visualization 5: Confusion Matrix
    ax5 = plt.subplot(5, 2, 5)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='d', ax=ax5)
    ax5.set_title('Confusion Matrix')
    #image_paths.append('static/confusion_matrix.png')

    # Visualization 6: Bar Chart - No. of employed vs unemployed customers
    # Count the number of employed (0) and unemployed (1) individuals
    employment_counts = df['self_employed'].value_counts().sort_index()

    # Create a bar chart for employed and unemployed people
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Employed', 'Unemployed'], employment_counts, color=['skyblue', 'lightcoral'])
    plt.xlabel('Employment Status')
    plt.ylabel('Count')
    plt.title('Distribution of Employment Status')

    # Annotate the bars with count values
    for bar, count in zip(bars, employment_counts):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height(), str(count), va='bottom')
    #image_paths.append('static/bar_chart2.png')

    # Visualization 7: Scatter Plot - CIBIL Score vs. Bank Asset Value
    ax7 = plt.subplot(5, 2, 7)
    sns.scatterplot(data=df, x='cibil_score', y='bank_asset_value', hue='loan_status', palette='Set3', s=50, ax=ax7)
    ax7.set_title('CIBIL Score vs. Bank Asset Value')
    #image_paths.append('static/scatter_plot3.png')

    # Visualization 8: Mean Cibil score for accepted and rejected customers
    # Calculate the mean CIBIL score for accepted and rejected loans
    mean_cibil_score_accepted = df[df['loan_status'] == 0]['cibil_score'].mean()
    mean_cibil_score_rejected = df[df['loan_status'] == 1]['cibil_score'].mean()

    # Create a bar chart to compare the mean CIBIL score for accepted and rejected loans
    plt.figure(figsize=(6, 6))
    plt.bar(['Accepted', 'Rejected'], [mean_cibil_score_accepted, mean_cibil_score_rejected], color=['green', 'red'])
    plt.xlabel('Loan Status')
    plt.ylabel('Mean CIBIL Score')
    plt.title('Mean CIBIL Score for Accepted vs. Rejected Loans')
    #image_paths.append('static/bar_chart4.png')

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

def split_dataframe(df, n_splits):
    splits = np.array_split(df, n_splits)
    return splits

def make_dataframe(split):
    df = pd.DataFrame(split, columns=data.columns)
    y = df['loan_status']
    X = df.drop('loan_status', axis=1)
    return X, y

def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def add_noise(X_test):
    laplace_scale = 0.1
    laplace_noise = np.random.laplace(loc=0, scale=laplace_scale, size=X_test.shape)
    noisy_data = X_test + laplace_noise
    return noisy_data

def model(X_train, y_train):
    svm_classifier = SVC()  # Create a Support Vector Machine classifier
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def cal_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

data = pd.read_csv('loan_approval.csv')  # Load your dataset here

@app.route('/pate')
def pate_analysis():
    num_splits = 5
    result_splits = split_dataframe(data, num_splits)
    accuracy = []
    teacher_model = []

    for i in result_splits:
        X, y = make_dataframe(i)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        teacher = model(X_train, y_train)
        teacher_model.append(teacher)
        y_pred = predict(teacher, X_test)
        acc = cal_accuracy(y_test, y_pred)
        accuracy.append(acc)

    return render_template('pate.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
