# app.py

# Import necessary libraries
from flask import Flask, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import io
import base64

app = Flask(__name__)

def run_model_and_generate_plots():
    """
    This function contains the entire machine learning pipeline from data generation
    to model training, evaluation, and visualization. It returns the results
    and the paths to the saved plot images.
    """
    
    np.random.seed(42)
    num_students = 100
    hours_studied = np.random.uniform(1, 20, num_students)
    attendance_percentage = np.random.uniform(50, 100, num_students)
    exam_score = (5 * hours_studied) + (0.2 * attendance_percentage) + np.random.normal(0, 5, num_students)
    exam_score = np.clip(exam_score, 0, 100)
    student_data = pd.DataFrame({
        'Hours_Studied': hours_studied,
        'Attendance_Percentage': attendance_percentage,
        'Exam_Score': exam_score
    })

    
    features = ['Hours_Studied', 'Attendance_Percentage']
    target = 'Exam_Score'
    X = student_data[features]
    y = student_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    evaluation_metrics = {
        "mse": f"{mse:.2f}",
        "rmse": f"{rmse:.2f}",
        "r2": f"{r2:.2f}"
    }

    new_student_study_hours = 15.5
    new_student_attendance = 95
    new_student_data = pd.DataFrame({
        'Hours_Studied': [new_student_study_hours],
        'Attendance_Percentage': [new_student_attendance]
    })
    predicted_score = model.predict(new_student_data)
    
    new_prediction = {
        "hours": new_student_study_hours,
        "attendance": new_student_attendance,
        "score": f"{predicted_score[0]:.2f}"
    }

    # 
    plt.style.use('fivethirtyeight')
    plot_urls = {}

    def plot_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig) 
        return f"data:image/png;base64,{plot_url}"

    
    fig1, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig1.suptitle('Exploratory Analysis of Student Data', fontsize=20, weight='bold')
    sns.regplot(ax=axes[0], x='Hours_Studied', y='Exam_Score', data=student_data, scatter_kws={'alpha':0.6, 's':50, 'edgecolor':'k'}, line_kws={'color':'#FF5733', 'linewidth': 3})
    axes[0].set_title('Hours Studied vs. Exam Score', fontsize=16)
    sns.regplot(ax=axes[1], x='Attendance_Percentage', y='Exam_Score', data=student_data, scatter_kws={'alpha':0.6, 's':50, 'edgecolor':'k', 'color': 'cornflowerblue'}, line_kws={'color':'#FF5733', 'linewidth': 3})
    axes[1].set_title('Attendance vs. Exam Score', fontsize=16)
    plot_urls['exploratory'] = plot_to_base64(fig1)

  
    fig2 = plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=60, color='limegreen')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '#FF5733', lw=3, linestyle='--')
    plt.title('Actual vs. Predicted Exam Scores', fontsize=18, weight='bold')
    plt.xlabel('Actual Scores', fontsize=14)
    plt.ylabel('Predicted Scores', fontsize=14)
    plot_urls['actual_vs_predicted'] = plot_to_base64(fig2)

    residuals = y_test - y_pred
    fig3 = plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k', s=60, color='orchid')
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='#FF5733', lw=3, linestyles='--')
    plt.title('Residual Plot', fontsize=18, weight='bold')
    plt.xlabel('Predicted Scores', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    plot_urls['residuals'] = plot_to_base64(fig3)
    
    return evaluation_metrics, new_prediction, plot_urls


@app.route('/')
def index():

    metrics, prediction, plots = run_model_and_generate_plots()
    
 
    return render_template('index.html', 
                           metrics=metrics, 
                           prediction=prediction, 
                           plots=plots)


if __name__ == '__main__':
 
    app.run(debug=True)
