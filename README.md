🏠 Real Estate Investment Advisor (Streamlit App)

A data-driven real estate analysis and investment advisory application built using Python, Pandas, and Streamlit.
The app performs data preparation, exploratory data analysis (EDA) and provides a rule-based investment recommendation for residential properties in India.

🚀 Project Features

📂 Upload and analyze real estate CSV data

🧹 Automated data cleaning & preprocessing

📊 Exploratory Data Analysis (EDA)

🏙️ City-wise median price analysis

🧠 Feature engineering (BHK, amenities, parking, age of property, etc.)

✅ Rule-based Good Investment classification

🖥️ Interactive Streamlit web application

🛠️ Tech Stack

Python 3

Pandas & NumPy

Streamlit

Matplotlib / Seaborn (for EDA)

Git & GitHub

📁 Project Structure
real_estate_project/
│
├── data/
│   └── india_housing_prices_cleaned.csv
│
├── data/src/
│   ├── data_prep.py        # Data cleaning & feature engineering
│   ├── eda_plots.py        # EDA visualizations
│   └── app/
│       ├── streamlit_app.py
│       └── requirements.txt
│
├── venv/
├── .gitignore
└── README.md

📊 Dataset

Source: Indian real estate housing data

Key Columns:

City, State

Size_in_SqFt

Price_in_Lakhs

BHK

Amenities

Parking_Space

Year_Built

⚠️ Raw data was cleaned and processed before analysis.

🧠 Investment Logic

A property is labeled as Good Investment (1) if it meets multiple criteria such as:

Price per sq. ft. below city median

Higher BHK count

Ready-to-move status

Availability of parking

Multiple amenities

Otherwise, it is marked as 0 (Not Recommended).

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/rajansonali/real_estate_project.git
cd real_estate_project

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r data/src/app/requirements.txt

4️⃣ Run Streamlit App
streamlit run data/src/app/streamlit_app.py

📌 Output

Interactive web dashboard

Investment recommendation per property

Cleaned and feature-engineered dataset

🎯 Use Cases

Real estate price analysis

Investment decision support

Data analytics portfolio project

Streamlit application demo

📈 Future Improvements

Machine learning price prediction

ROI forecasting

City-wise demand trends

Deployment on Streamlit Cloud
