Student Score Prediction Model
A web-based machine learning project that predicts student exam scores from study hours and attendance. It features a full data analysis, model evaluation, and visualizes the results on a professional webpage.

✨ Features
Data Analysis: Generates and describes a synthetic dataset for student performance.

Machine Learning Model: Implements a Linear Regression model to predict exam scores.

In-Depth Evaluation: Calculates and displays key performance metrics like R-Squared, MSE, and RMSE.

Rich Visualizations: Generates multiple plots to analyze the data and model performance, including:

Exploratory scatter plots.

An "Actual vs. Predicted" plot to gauge accuracy.

A residual plot for diagnostic checking.

Web Interface: Presents all findings in a clean, user-friendly webpage built with Flask and Tailwind CSS.

🛠️ Tech Stack
Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NumPy

Data Visualization: Matplotlib, Seaborn

Frontend: HTML, Tailwind CSS

Deployment: Gunicorn, Render

To set up and run the project on your local machine.

Prerequisites
Python 3.8 or higher

pip (Python package installer)

Installation & Setup
Clone the repository:

git clone https://github.com/bits-blitzz/guvi_iitp_p2.git

Navigate to the project directory:

cd your-repository-name

Install the required dependencies:
It's recommended to use a virtual environment to keep dependencies isolated.

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install packages from requirements.txt
pip install -r requirements.txt

Running the Project
Start the Flask application:

flask run

You can run python app.py.

View the application:
Open your web browser and navigate to http://127.0.0.1:5000. You should see the project's homepage with all the analysis and graphs.

🌐 Deployment
This application is deployed as a Web Service on Render. The live version can be accessed at the URL provided at the top of this README.

The deployment is configured to automatically build and launch the application based on the requirements.txt and gunicorn app:app start command.

📂 Repository Structure
.
├── app.py                  # The main Flask application file
├── requirements.txt        # Lists all Python dependencies
├── templates/
│   └── index.html          # The HTML template for the webpage
└── .gitignore              # Specifies files for Git to ignore
└── README.md               # This file
