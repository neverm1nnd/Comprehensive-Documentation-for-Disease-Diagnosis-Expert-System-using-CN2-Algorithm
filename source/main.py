import streamlit as st
import pandas as pd
import pickle
import os
import logging
from CN2 import CN2
import numpy as np

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'best_cn2_model.pkl'
PROCESSED_DATA_PATH = '../Data/dataset_processed.csv'
TRAIN_DATA_PATH = '../Data/dataset_train.csv'
TEST_DATA_PATH = '../Data/dataset_test.csv'

st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Основные цвета */
    :root {
        --dark-blue-bg: #0a192f;
        --teal-accent: #00b4d8;
        --light-teal: #90e0ef;
        --text-color: #e6f1ff;
    }

    .stApp, .main {
        background-color: var(--dark-blue-bg) !important;
        color: var(--text-color) !important;
        font-family: 'Arial', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--teal-accent) !important;
    }

    [data-testid="stSidebar"] {
        background-color: #112240 !important;
    }

    .stButton>button {
        background-color: var(--teal-accent) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background-color: var(--light-teal) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 180, 216, 0.3) !important;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #112240 !important;
        color: var(--text-color) !important;
        border: 1px solid var(--teal-accent) !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }

    .stCheckbox, .stRadio {
        color: var(--text-color) !important;
    }

    .dataframe {
        background-color: #112240 !important;
        color: var(--text-color) !important;
    }

    .dataframe th {
        background-color: var(--teal-accent) !important;
        color: #000 !important;
    }

    .dataframe tr:nth-child(even) {
        background-color: #0e1a2e !important;
    }

    .stTabs [role="tablist"] button {
        color: var(--text-color) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--teal-accent) !important;
        border-bottom: 2px solid var(--teal-accent) !important;
    }
</style>
""", unsafe_allow_html=True)

def train_and_save_model():
    try:
        if not all(os.path.exists(path) for path in [PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH]):
            from preprocessing import preprocess_data
            preprocess_data()

        cn2 = CN2(TRAIN_DATA_PATH, TEST_DATA_PATH)
        rules = cn2.fit_CN2()
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump((cn2, rules), f)
        return cn2, rules
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        st.error(f"Error training model: {str(e)}")
        return None, None

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                cn2, rules = pickle.load(f)
            return cn2, rules
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
    return None, None

def get_symptoms_list():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return [col for col in df.columns if col != 'class']
    except Exception as e:
        logging.error(f"Error loading symptoms list: {str(e)}")
        st.error(f"Error loading symptoms list: {str(e)}")
        return []

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0  
    return dot_product / (norm_a * norm_b)

def manual_knn(X_train, y_train, X_test, k=5):
    predictions = []
    for test_sample in X_test:
        similarities = [cosine_similarity(test_sample, train_sample) for train_sample in X_train]
        k_indices = np.argsort(similarities)[-k:][::-1] 
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(most_common)
    return predictions

def predict_disease(symptoms):
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        X = df.drop(columns=['class']).values  
        y = df['class'].values
        
        input_data = pd.DataFrame(0, index=[0], columns=df.columns.drop('class'))
        for symptom in symptoms:
            if symptom in input_data.columns:
                input_data[symptom] = 1
            else:
                st.warning(f"Symptom '{symptom}' not found in the dataset.")
        X_test = input_data.values
        
        prediction = manual_knn(X, y, X_test, k=5)[0]
        
        return prediction, None, None  

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "An error occurred", None, None

def main():
    # Initialize session state
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []

    st.title("Medical Diagnosis System")
    st.write("Select your symptoms and get a preliminary diagnosis. **Warning:** This system is for educational purposes only.")

    # Load/train model
    cn2, rules = load_model()
    if cn2 is None or rules is None:
        with st.spinner("Training model... This may take a few minutes..."):
            cn2, rules = train_and_save_model()
        if cn2 is not None and rules is not None:
            st.success("Model trained successfully!")
        else:
            st.error("Model training failed. Check data files.")
            return

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Select Symptoms")
        # Diagnosis button
        if st.button("Make Diagnosis", use_container_width=True, key="diagnosis_btn_top"):
            if not st.session_state.selected_symptoms:
                st.warning("Please select at least one symptom")
            else:
                with col2:
                    st.subheader("Diagnosis Results")
                    prediction, _, _ = predict_disease(st.session_state.selected_symptoms)
                    if "error" in prediction.lower():
                        st.error(prediction)
                    else:
                        st.success(f"Preliminary diagnosis: **{prediction}**")
                        st.write("Please consult a healthcare professional for confirmation")

        symptoms_list = get_symptoms_list()
        if not symptoms_list:
            st.error("Failed to load symptoms list.")
            return

        search_query = st.text_input("Search symptoms", "", key="symptom_search")

        filtered_symptoms = [s for s in symptoms_list if search_query.lower() in s.lower()]

        st.markdown("""
        <style>
            .symptoms-container {
                max-height: 65vh;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #00b4d8;
                border-radius: 8px;
                margin: 10px 0;
            }
        </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="symptoms-container">', unsafe_allow_html=True)
            
            if filtered_symptoms:
                cols = st.columns(4)
                chunk_size = (len(filtered_symptoms) // 4) + 1
                
                for i in range(4):
                    with cols[i]:
                        start = i * chunk_size
                        end = (i+1) * chunk_size
                        for symptom in filtered_symptoms[start:end]:
                            # Generate unique key using symptom name hash
                            symptom_hash = hash(symptom) & 0xFFFFFFFF
                            is_checked = st.checkbox(
                                symptom,
                                key=f"cb_{symptom_hash}",
                                value=symptom in st.session_state.selected_symptoms,
                                on_change=lambda s=symptom: update_selected_symptoms(s)
                            )
            
            else:
                st.warning("No symptoms found")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if not st.session_state.selected_symptoms:
            st.subheader("Instructions")
            st.markdown("""
            1. Select symptoms from the left panel
            2. Click 'Make Diagnosis' button
            3. Results will appear here
            """)
            # Уменьшаем изображение
            
        else:
            # Добавляем отступ для результатов
            st.markdown("""
            <style>
                .diagnosis-results {
                    margin-left: 20px;
                    padding-left: 20px;
                    border-left: 2px solid #00b4d8;
                }
            </style>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="diagnosis-results">', unsafe_allow_html=True)
                # ... [код вывода диагноза] ...
                st.markdown('</div>', unsafe_allow_html=True)

def update_selected_symptoms(symptom):
    """Callback function to update selected symptoms"""
    if symptom in st.session_state.selected_symptoms:
        st.session_state.selected_symptoms.remove(symptom)
    else:
        st.session_state.selected_symptoms.append(symptom)

if __name__ == "__main__":
    main()