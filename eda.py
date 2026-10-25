
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder to save graphs
os.makedirs("graphs", exist_ok=True)

# Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

sns.set_theme(style="whitegrid")

# GRAPH 1: Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("graphs/1_churn_distribution.png")
plt.show()

# GRAPH 2: Model Comparison
if os.path.exists("model_comparison.csv"):
    results = pd.read_csv("model_comparison.csv")
    results_long = results.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1 Score (%)"
        ],
        var_name="Metric",
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_long,
        x="Metric",
        y="Score",
        hue="Model"
    )
    plt.title("Comparison of Machine Learning Models")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("graphs/2_model_comparison.png")
    plt.show()
else:
    print("model_comparison.csv not found.")

# GRAPH 3: Tenure and Churn
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=20,
    multiple="stack"
)
plt.title("Customer Tenure and Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("graphs/3_tenure_churn.png")
plt.show()

# GRAPH 4: Monthly Charges and Churn
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("graphs/4_monthly_charges_churn.png")
plt.show()

print("\nAll graphs created successfully!")
print("Check the graphs folder in your project.")
