# -Data-Cleaning-Preprocessing
🧹 Task 1: Data Cleaning & Preprocessing — Titanic Dataset
# 🎯 Objective

Learn how to clean and prepare raw data for Machine Learning by handling missing values, encoding categorical variables, normalizing features, and detecting/removing outliers.

# 🧰 Tools & Libraries

Python 3.x

Pandas – Data manipulation and cleaning

NumPy – Numerical computations

Matplotlib / Seaborn – Data visualization

scikit-learn – Feature scaling (StandardScaler)

# 📂 Dataset

Name: Titanic-Dataset.csv
You can download it from: Titanic Dataset

or use the provided file.

# ⚙️ Steps in the Code
1️⃣ Import Libraries

Imports required Python libraries (pandas, numpy, seaborn, matplotlib).

2️⃣ Load Dataset

Reads the Titanic dataset into a pandas DataFrame:

df = pd.read_csv("Titanic-Dataset.csv")

3️⃣ Explore Data

Displays key information, such as:

Number of rows and columns

Data types of each column

Count of missing values

Summary statistics

4️⃣ Handle Missing Values

Replaces missing values with median or mode to maintain data consistency:

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.dropna(thresh=len(df.columns) - 3)

5️⃣ Encode Categorical Variables

Converts categorical features into numerical using One-Hot Encoding:

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

6️⃣ Standardize Numerical Features

Scales numerical values to have zero mean and unit variance using StandardScaler:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

7️⃣ Visualize Outliers

Uses boxplots to detect outliers:

sns.boxplot(data=df_encoded[numeric_cols])
plt.show()

8️⃣ Remove Outliers (IQR Method)

Removes extreme values based on the Interquartile Range:

def remove_outliers_iqr(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

9️⃣ Save Cleaned Data

Exports the cleaned dataset:

df_cleaned.to_csv("Titanic_Cleaned.csv", index=False)

# 📊 Outputs

Titanic_Cleaned.csv – The final cleaned and preprocessed dataset.

Boxplots before and after outlier removal.

Console output showing dataset info and cleaning progress.

# 🧠 Key Learnings

Handling missing values using imputation (mean, median, mode).

Performing categorical encoding for machine learning models.

Standardizing numeric features for consistent scaling.

Detecting and removing outliers to improve data quality.

Understanding data preprocessing workflow before modeling.

# ▶️ How to Run

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Place the dataset (Titanic-Dataset.csv) in your working directory.

Run the Python script in Jupyter Notebook, VS Code, or any Python IDE.

Check the generated file Titanic_Cleaned.csv.
