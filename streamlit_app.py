import streamlit as st
from streamlit_option_menu import option_menu
import chardet
import tempfile
from CLASSIFICATION import *

# Initialize session state
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None
classificationList = ["setup_cls","create_cls_df", "create_cls","predict_cls","predict_cls_df", "ensemble_cls","ensemble_cls_df", "tune_cls","tune_cls_df", "save_create_cls", "save_enseble_cls", "save_tune_cls"]
for i in classificationList:
    if i not in st.session_state:
        st.session_state[i] = None
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    df = None

    if uploaded_file is not None:
        # Detect encoding using chardet
        raw_data = uploaded_file.read(1024)
        detected_encoding = chardet.detect(raw_data).get('encoding', 'utf-8')
        st.write(f"Detected Encoding: {detected_encoding}")
        uploaded_file.seek(0)

        # Attempt to load the file with detected encoding, falling back if necessary
        try:
            st.session_state.dataset = pd.read_csv(uploaded_file, encoding=detected_encoding)
            st.write("File loaded successfully!")
        except (UnicodeDecodeError, pd.errors.ParserError):
            st.warning(f"Failed to load file with encoding '{detected_encoding}'. Trying 'ISO-8859-1'.")
            try:
                st.session_state.dataset = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                st.write("File loaded successfully with ISO-8859-1 encoding!")
            except Exception as e:
                st.error("Unable to load file. Please check the file format and encoding.")
                st.error(f"Error details: {e}")

        # Check if DataFrame is empty after loading
        if df is not None and df.empty:
            st.warning("The uploaded file is empty or only contains headers. Please upload a valid CSV file.")
            st.session_state.dataset = None  # Reset if empty

    # Option menu for Machine Learning choices
    ml_menu = option_menu(
        "Machine Learning",
        options=["Regression", "Time-series", "Classification", "Clustering", "Anomaly-Detection", "Predictions"],
        default_index=0,
        menu_icon="robot",
    )

if ml_menu == "Regression":
    pass
elif ml_menu == "Time-series":
    pass
elif ml_menu == "Classification":
    instance = Classification(st.session_state.dataset)
    instance.runApp()
