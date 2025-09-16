import pandas as pd
import numpy as np
import os

DATA_PATH = '../Data/dataset.csv'
def manual_train_test_split(df, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split_index = int(len(df) * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    train = df.iloc[train_indices]
    test = df.iloc[test_indices]
    return train, test
def preprocess_data():
    df = pd.read_csv(DATA_PATH)
    print(df.head())

    if 'Disease' in df.columns:
        df = df.rename(columns={'Disease': 'class'})
    else:
        raise ValueError("Dataset must contain a 'Disease' column")

    symptom_cols = [col for col in df.columns if 'Symptom_' in col]
    
    if not symptom_cols:
        raise ValueError("No symptom columns found (expected columns with 'Symptom_' prefix)")

    symptoms = set()
    for col in symptom_cols:
        symptoms.update(df[col].dropna().unique())
    symptoms = [s for s in symptoms if pd.notna(s)]  

    for symptom in symptoms:
        df[symptom] = 0
        for col in symptom_cols:
            df[symptom] = df[symptom] | (df[col] == symptom).astype(int)

    df = df.drop(columns=symptom_cols)

    os.makedirs(os.path.dirname('../Data/'), exist_ok=True)

    df.to_csv('../Data/dataset_processed.csv', index=False)


    train, test = manual_train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv('../Data/dataset_train.csv', index=False)
    test.to_csv('../Data/dataset_test.csv', index=False)

    
    return df

if __name__ == '__main__':
    preprocess_data()