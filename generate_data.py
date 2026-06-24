# generate_data.py
import pandas as pd
import numpy as np
import os

def create_dataset(n_samples, drift=False):
    np.random.seed(42 if not drift else 2026)
    age = np.random.randint(18, 70, size=n_samples)
    if not drift:
        # Normal baseline conditions
        income = np.random.normal(55000, 15000, size=n_samples).round(2)
        credit_score = np.random.randint(600, 850, size=n_samples)
        employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], size=n_samples, p=[0.7, 0.25, 0.05])
    else:
        # Drifted conditions: Incomes drop, credit scores tank, unemployment spikes
        income = np.random.normal(42000, 12000, size=n_samples).round(2)
        credit_score = np.random.randint(500, 750, size=n_samples)
        employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], size=n_samples, p=[0.5, 0.3, 0.2])
    risk_score = (850 - credit_score) * 0.6 + (60000 - income) * 0.005
    risk_score += np.random.normal(0, 20, size=n_samples) # add some noise
    target = (risk_score > 180).astype(int)
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'employment_type': employment_type,
        'default': target
    })
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    train_df = create_dataset(2000, drift=False)
    train_df.to_csv('data/baseline_training.csv', index=False)
    prod_df = create_dataset(1000, drift=True)
    prod_df.to_csv('data/production_simulation.csv', index=False)
    print("✅ Synthetic datasets created successfully inside the 'data/' folder!")