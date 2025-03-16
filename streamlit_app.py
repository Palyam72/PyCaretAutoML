## streamlit_app.py
import streamlit as st
from streamlit_option_menu import option_menu
import chardet
import pandas as pd
from CLASSIFICATION import Classification
from REGRESSION import Regression
from CLUSTERING import Clustering

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        raw_data = uploaded_file.read(1024)
        detected_encoding = chardet.detect(raw_data).get('encoding', 'utf-8')
        st.write(f"Detected Encoding: {detected_encoding}")
        uploaded_file.seek(0)

        try:
            st.session_state.dataset = pd.read_csv(uploaded_file, encoding=detected_encoding)
            st.success("File loaded successfully!")
        except Exception as e:
            st.error("Failed to load file. Check the format and encoding.")
            st.error(str(e))

    ml_menu = option_menu(
        "Machine Learning",
        options=["Regression", "Classification", "Clustering"],
        default_index=0,
        menu_icon="robot",
    )

if ml_menu == "Classification" and st.session_state.dataset is not None:
    classification_instance = Classification(st.session_state.dataset)
    classification_instance.run_app()
elif ml_menu == "Regression" and st.session_state.dataset is not None:
    regression_instance = Regression(st.session_state.dataset)
    regression_instance.run_app()
elif ml_menu == "Clustering":
    clustering_instance = Clustering(st.session_state.dataset)
    clustering_instance.run_app()
    
else:
    st.info("Please upload a dataset to proceed")
