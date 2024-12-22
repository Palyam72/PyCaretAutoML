## classification.py
import streamlit as st
import pandas as pd
from pycaret.classification import *

# Initialize session state for classification-related variables
classification_keys = [
    "setup_cls", "compare_cls", "create_cls", "create_cls_df", "predict_cls", 
    "predict_cls_df", "ensemble_cls", "ensemble_cls_df", "tune_cls", "tune_cls_df"
]
for key in classification_keys:
    if key not in st.session_state:
        st.session_state[key] = None

class Classification:
    def __init__(self, dataset):
        self.dataset = dataset
        self.models_list = [
            'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 
            'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost'
        ]

    def select_target(self):
        if st.session_state.setup_cls is None:
            st.subheader("Set up your environment")
            st.divider()
            target = st.selectbox("Select the target column", self.dataset.columns)
            if target:
                st.session_state.setup_cls = setup(data=self.dataset, target=target)
                st.session_state.setup_summary = pull()
                st.dataframe(st.session_state.setup_summary)
        else:
            st.subheader("Environment already set up")
            st.dataframe(st.session_state.setup_summary)

    def compare_models(self):
        if st.session_state.compare_cls is None and st.session_state.setup_cls is not None:
            st.session_state.compare_cls = compare_models()
            st.session_state.compare_summary = pull()
            st.dataframe(st.session_state.compare_summary)
        elif st.session_state.compare_cls is not None:
            st.markdown("Models comparison already performed")
            st.dataframe(st.session_state.compare_summary)
        else:
            st.warning("Set up your environment first")

    def create_model(self):
        self.select_target()
        self.compare_models()
        if st.session_state.create_cls is None and st.session_state.setup_cls is not None:
            selected_model = st.selectbox("Choose a model to create", self.models_list)
            if selected_model:
                st.session_state.create_cls = create_model(selected_model)
                st.session_state.create_cls_df = pull()
                st.dataframe(st.session_state.create_cls_df)

                st.session_state.predict_cls = predict_model(st.session_state.create_cls)
                st.session_state.predict_cls_df = pull()
                st.dataframe(st.session_state.predict_cls_df)
        elif st.session_state.create_cls is not None:
            st.info("Model already created")
            st.dataframe(st.session_state.create_cls_df)
            st.dataframe(st.session_state.predict_cls_df)

    def ensemble_model(self):
        if st.session_state.ensemble_cls is None and st.session_state.create_cls is not None:
            st.session_state.ensemble_cls = ensemble_model(st.session_state.create_cls)
            st.session_state.ensemble_cls_df = pull()
            st.dataframe(st.session_state.ensemble_cls_df)
        elif st.session_state.ensemble_cls is not None:
            st.info("Ensemble model already created")
            st.dataframe(st.session_state.ensemble_cls_df)
        else:
            st.warning("Create a base model first")

    def tune_model(self):
        if st.session_state.tune_cls is None and st.session_state.create_cls is not None:
            st.session_state.tune_cls = tune_model(st.session_state.create_cls)
            st.session_state.tune_cls_df = pull()
            st.dataframe(st.session_state.tune_cls_df)
        elif st.session_state.tune_cls is not None:
            st.info("Model already tuned")
            st.dataframe(st.session_state.tune_cls_df)
        else:
            st.warning("Create a base model first")

    def run_app(self):
        col1, col2 = st.columns([1, 2])

        with col1:
            option = st.radio("Choose an action", ["Create Model", "Ensemble Model", "Tune Model"])

        with col2:
            if option == "Create Model":
                self.create_model()
            elif option == "Ensemble Model":
                self.ensemble_model()
            elif option == "Tune Model":
                self.tune_model()
