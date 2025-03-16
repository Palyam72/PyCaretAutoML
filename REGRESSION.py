## classification.py
import streamlit as st
import pandas as pd
from pycaret.regression import *

# Initialize session state for classification-related variables
regression_keys = [
    "setup_cls", "compare_cls", "create_cls", "create_cls_df", "predict_cls", 
    "predict_cls_df", "ensemble_cls", "ensemble_cls_df", "tune_cls", "tune_cls_df", "tune_cls_df","tune_ensembled_cls","tune_ensembled_cls_df", "save_create_cls",
    "save_ensemble_cls", "save_tune_cls", "save_ensembled_tune_cls", "predict_ensemble_cls", "predict_ensemble_cls_df", "predict_tune_cls",
    "predict_tune_cls_df","predict_ensembled_tune_cls","predict_ensembled_tune_cls_df"
]
regression_keys=[x.replace("_cls","_reg") for x in regression_keys]
for key in regression_keys:
    if key not in st.session_state:
        st.session_state[key] = None

class Regression:
    def __init__(self, dataset):
        self.dataset = dataset
        self.models_list = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost']

    def select_target(self):
        if st.session_state.setup_reg is None:
            st.subheader("Set up your environment")
            st.divider()
            target = st.pills("Select the target column", self.dataset.columns)
            if target:
                st.session_state.setup_reg = setup(data=self.dataset, target=target)
                st.session_state.setup_summary = pull()
                st.dataframe(st.session_state.setup_summary)
        else:
            st.subheader("Environment already set up")
            st.dataframe(st.session_state.setup_summary)

    def compare_models(self):
        if st.session_state.compare_reg is None and st.session_state.setup_reg is not None:
            st.session_state.compare_reg = compare_models()
            st.session_state.compare_summary = pull()
            st.dataframe(st.session_state.compare_summary)
        elif st.session_state.compare_reg is not None:
            st.markdown("Models comparison already performed")
            st.dataframe(st.session_state.compare_summary)
        else:
            st.warning("Set up your environment first")

    def create_model(self):
        self.select_target()
        self.compare_models()
        if st.session_state.create_reg is None and st.session_state.setup_reg is not None:
            selected_model = st.pills("Choose a model to create", self.models_list)
            if selected_model:
                st.session_state.create_reg = create_model(selected_model)
                st.session_state.create_reg_df = pull()
                st.dataframe(st.session_state.create_reg_df)

                st.session_state.predict_reg = predict_model(st.session_state.create_reg)
                st.session_state.predict_reg_df = pull()
                st.dataframe(st.session_state.predict_reg_df)
        elif st.session_state.create_reg is not None:
            st.info("Model already created")
            st.dataframe(st.session_state.create_reg_df)
            st.dataframe(st.session_state.predict_reg_df)

    def ensemble_model(self):
        if st.session_state.ensemble_reg is None and st.session_state.create_reg is not None:
            st.session_state.ensemble_reg = ensemble_model(st.session_state.create_reg)
            st.session_state.ensemble_reg_df = pull()
            st.dataframe(st.session_state.ensemble_reg_df)
            st.session_state.predict_ensemble_reg = predict_model(st.session_state.ensemble_reg)
            st.session_state.predict_ensemble_reg_df = pull()
            st.dataframe(st.session_state.predict_ensemble_reg_df)
        elif st.session_state.ensemble_reg is not None:
            st.info("Ensemble model already created")
            st.dataframe(st.session_state.ensemble_reg_df)
            st.dataframe(st.session_state.predict_ensemble_reg_df)
        else:
            st.warning("Create a base model first")

    def tune_model(self):
        if st.session_state.tune_reg is None and st.session_state.create_reg is not None:
            st.session_state.tune_reg = tune_model(st.session_state.create_reg)
            st.session_state.tune_reg_df = pull()
            st.dataframe(st.session_state.tune_reg_df)
            st.session_state.predict_tune_reg = predict_model(st.session_state.tune_reg)
            st.session_state.predict_tune_reg_df = pull()
            st.dataframe(st.session_state.predict_tune_reg_df)
        elif st.session_state.tune_reg is not None:
            st.info("Model already tuned")
            st.dataframe(st.session_state.tune_reg_df)
            st.dataframe(st.session_state.predict_tune_reg_df)
        else:
            st.warning("Create a base model first")
    def tune_ensemble_model(self):
        if st.session_state.tune_ensembled_reg is None:
            st.session_state.tune_ensembled_reg = tune_model(st.session_state.ensemble_reg)
            st.session_state.tune_ensembled_reg_df = pull()
            st.dataframe(st.session_state.tune_ensembled_reg_df)
            st.session_state.predict_tune_ensemble_reg = predict_model(st.session_state.tune_ensembled_reg)
            st.session_state.predict_tune_ensemble_reg_df = pull()
            st.dataframe(st.session_state.predict_tune_ensemble_reg_df)
        elif st.session_state.tune_ensembled_reg is not None:
            st.info("Ensemble model already tuned")
            st.dataframe(st.session_state.tune_ensembled_reg_df)
            st.dataframe(st.session_state.predict_tune_ensemble_reg_df)
        else:
            st.warning("Create an ensemble model first")
    
    def save_models(self):
        choose_model_to_save = st.pills(
            "Choose the model to save", 
            ["Normal Model", "Ensembled Model", "Tuned Normal Model", "Tuned Ensembled Model"]
        )
        if choose_model_to_save == "Normal Model":
            if st.session_state.create_reg is not None:
                if st.toggle("Provide a custom name for the model"):
                    name_model = st.text_input("Enter model name")
                    if name_model:
                        st.session_state.save_create_reg = save_model(st.session_state.create_reg, name_model)
                        st.success(f"Model saved as {name_model}.")
                        st.download_button(
                            label="Download Model",
                            data=open(f"{name_model}.pkl", "rb").read(),
                            file_name=f"{name_model}.pkl"
                        )
                else:
                    pn = st.pills("Proceed to download Model or not",[True, False])
                    if pn:
                        st.session_state.save_create_reg = save_model(st.session_state.create_reg, "normal model")
                        st.success(f"Model saved as 'normal model'.")
                        st.download_button(
                            label="Download Model",
                            data=open(f"{name_model}.pkl", "rb").read(),
                            file_name=f"{name_model}.pkl"
                        )

            else:
                st.warning("No model available to save.")
        elif choose_model_to_save == "Ensembled Model":
            if st.session_state.ensemble_reg is not None:
                if st.toggle("Provide a custom name for the ensemble model"):
                    name_ensemble = st.text_input("Enter ensemble model name")
                    if name_ensemble:
                        st.session_state.save_ensemble_reg = save_model(st.session_state.ensemble_reg, name_ensemble)
                        st.success(f"Ensemble model saved as {name_ensemble}.")
                        st.download_button(
                            label="Download Ensemble Model",
                            data=open(f"{name_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_ensemble}.pkl"
                        )
                else:
                    pn = st.pills("Proceed to download Model or not",[True, False])
                    if pn:
                        name_ensemle="ensmble model"
                        st.session_state.save_ensemble_reg = save_model(st.session_state.ensemble_reg, name_ensemble)
                        st.success(f"Ensemble model saved as {name_ensemble}.")
                        st.download_button(
                            label="Download Ensemble Model",
                            data=open(f"{name_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_ensemble}.pkl"
                        )
            else:
                st.warning("No ensemble model available to save.")
        elif choose_model_to_save == "Tuned Normal Model":
            if st.session_state.tune_reg is not None:
                if st.toggle("Provide a custom name for the tuned model"):
                    name_tune = st.text_input("Enter tuned model name")
                    if name_tune:
                        st.session_state.save_tune_reg = save_model(st.session_state.tune_reg, name_tune)
                        st.success(f"Tuned model saved as {name_tune}.")
                        st.download_button(
                            label="Download Tuned Model",
                            data=open(f"{name_tune}.pkl", "rb").read(),
                            file_name=f"{name_tune}.pkl"
                        )
                else:
                    pn = st.pills("Proceed to download Model or not",[True, False])
                    if pn:
                        name_tune = "tuned model"
                        st.session_state.save_tune_reg = save_model(st.session_state.tune_reg, name_tune)
                        st.success(f"Tuned model saved as {name_tune}.")
                        st.download_button(
                            label="Download Tuned Model",
                            data=open(f"{name_tune}.pkl", "rb").read(),
                            file_name=f"{name_tune}.pkl"
                        )
                        
            else:
                st.warning("No tuned model available to save.")
        elif choose_model_to_save == "Tuned Ensembled Model":
            if st.session_state.tune_ensembled_reg is not None:
                if st.toggle("Provide a custom name for the tuned ensemble model"):
                    name_tune_ensemble = st.text_input("Enter tuned ensemble model name")
                    if name_tune_ensemble:
                        st.session_state.save_ensembled_tune_reg = save_model(st.session_state.tune_ensembled_reg, name_tune_ensemble)
                        st.success(f"Tuned ensemble model saved as {name_tune_ensemble}.")
                        st.download_button(
                            label="Download Tuned Ensemble Model",
                            data=open(f"{name_tune_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_tune_ensemble}.pkl"
                        )
                else:
                    pn = st.pills("Proceed to download Model or not",[True, False])
                    if pn:
                        name_tune_ensemble = "tuned ensemble model"
                        st.session_state.save_ensembled_tune_reg = save_model(st.session_state.tune_ensembled_reg, name_tune_ensemble)
                        st.success(f"Tuned ensemble model saved as {name_tune_ensemble}.")
                        st.download_button(
                            label="Download Tuned Ensemble Model",
                            data=open(f"{name_tune_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_tune_ensemble}.pkl"
                        )             
            else:
                st.warning("No tuned ensemble model available to save.")

                    
        
    def reCreate(self):
        st.session_state.create_reg = None
        st.session_state.create_reg_df = None
        st.session_state.predict_reg = None
        st.session_state.predict_reg_df = None
        st.session_state.ensemble_reg = None
        st.session_state.ensemble_reg_df = None
        st.session_state.tune_reg = None
        st.session_state.tune_reg_df = None
        st.warning("You wanted to recreate model so entire history have been deleted.")
        self.create_model()


    def run_app(self):
        col1, col2 = st.columns([1, 2])

        with col1:
            option = st.radio("Choose an action", ["Create Model", "Ensemble Model", "Tune Model","Tune Ensembled Model","Recreate Model","save your model"])

        with col2:
            if option == "Create Model":
                self.create_model()
            elif option == "Ensemble Model":
                self.ensemble_model()
            elif option == "Tune Model":
                self.tune_model()
            elif option == "Tune Ensembled Model":
                self.tune_ensemble_model()
            elif option == "Recreate Model":
                self.reCreate()
            elif option == "save your model":
                self.save_models()