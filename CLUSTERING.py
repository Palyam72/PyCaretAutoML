## classification.py
import streamlit as st
import pandas as pd
from pycaret.clustering import *

# Initialize session state for classification-related variables
regression_keys = [
    "setup_cls", "create_cls", "create_cls_df", "predict_cls", 
    "predict_cls_df", "ensemble_cls", "ensemble_cls_df", "tune_cls", "tune_cls_df", "tune_cls_df","tune_ensembled_cls","tune_ensembled_cls_df", "save_create_cls",
    "save_ensemble_cls", "save_tune_cls", "save_ensembled_tune_cls", "predict_ensemble_cls", "predict_ensemble_cls_df", "predict_tune_cls",
    "predict_tune_cls_df","predict_ensembled_tune_cls","predict_ensembled_tune_cls_df"
]
regression_keys=[x.replace("_cls","_clus") for x in regression_keys]
for key in regression_keys:
    if key not in st.session_state:
        st.session_state[key] = None

class Clustering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.models_list = ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes']

    def select_target(self):
        if st.session_state.setup_clus is None:
            st.subheader("Set up your environment")
            st.divider()
            st.session_state.setup_clus = setup(data=self.dataset)
            st.session_state.setup_summary = pull()
            st.dataframe(st.session_state.setup_summary)
        else:
            st.subheader("Environment already set up")
            st.dataframe(st.session_state.setup_summary)


    def create_model(self):
        self.select_target()
        if st.session_state.create_clus is None and st.session_state.setup_clus is not None:
            selected_model = st.pills("Choose a model to create", self.models_list)
            if selected_model:
                st.session_state.create_clus = create_model(selected_model)
                st.session_state.create_clus_df = pull()
                st.dataframe(st.session_state.create_clus_df)

                st.session_state.predict_clus = predict_model(st.session_state.create_clus,self.dataset)
                st.session_state.predict_clus_df = pull()
                st.dataframe(st.session_state.predict_clus_df)
        elif st.session_state.create_clus is not None:
            st.info("Model already created")
            st.dataframe(st.session_state.create_clus_df)
            st.dataframe(st.session_state.predict_clus_df)

    def ensemble_model(self):
        if st.session_state.ensemble_clus is None and st.session_state.create_clus is not None:
            st.session_state.ensemble_clus = ensemble_model(st.session_state.create_clus)
            st.session_state.ensemble_clus_df = pull()
            st.dataframe(st.session_state.ensemble_clus_df)
            st.session_state.predict_ensemble_clus = predict_model(st.session_state.ensemble_clus,self.dataset)
            st.session_state.predict_ensemble_clus_df = pull()
            st.dataframe(st.session_state.predict_ensemble_clus_df)
        elif st.session_state.ensemble_clus is not None:
            st.info("Ensemble model already created")
            st.dataframe(st.session_state.ensemble_clus_df)
            st.dataframe(st.session_state.predict_ensemble_clus_df)
        else:
            st.warning("Create a base model first")

    def tune_model(self):
        if st.session_state.tune_clus is None and st.session_state.create_clus is not None:
            st.session_state.tune_clus = tune_model(st.session_state.create_clus)
            st.session_state.tune_clus_df = pull()
            st.dataframe(st.session_state.tune_clus_df)
            st.session_state.predict_tune_clus = predict_model(st.session_state.tune_clus,self.dataset)
            st.session_state.predict_tune_clus_df = pull()
            st.dataframe(st.session_state.predict_tune_clus_df)
        elif st.session_state.tune_clus is not None:
            st.info("Model already tuned")
            st.dataframe(st.session_state.tune_clus_df)
            st.dataframe(st.session_state.predict_tune_clus_df)
        else:
            st.warning("Create a base model first")
    def tune_ensemble_model(self):
        if st.session_state.tune_ensembled_clus is None:
            st.session_state.tune_ensembled_clus = tune_model(st.session_state.ensemble_clus)
            st.session_state.tune_ensembled_clus_df = pull()
            st.dataframe(st.session_state.tune_ensembled_clus_df)
            st.session_state.predict_tune_ensemble_clus = predict_model(st.session_state.tune_ensembled_clus,self.dataset)
            st.session_state.predict_tune_ensemble_clus_df = pull()
            st.dataframe(st.session_state.predict_tune_ensemble_clus_df)
        elif st.session_state.tune_ensembled_clus is not None:
            st.info("Ensemble model already tuned")
            st.dataframe(st.session_state.tune_ensembled_clus_df)
            st.dataframe(st.session_state.predict_tune_ensemble_clus_df)
        else:
            st.warning("Create an ensemble model first")
    
    def save_models(self):
        choose_model_to_save = st.pills(
            "Choose the model to save", 
            ["Normal Model", "Ensembled Model", "Tuned Normal Model", "Tuned Ensembled Model"]
        )
        if choose_model_to_save == "Normal Model":
            if st.session_state.create_clus is not None:
                if st.toggle("Provide a custom name for the model"):
                    name_model = st.text_input("Enter model name")
                    if name_model:
                        st.session_state.save_create_clus = save_model(st.session_state.create_clus, name_model)
                        st.success(f"Model saved as {name_model}.")
                        st.download_button(
                            label="Download Model",
                            data=open(f"{name_model}.pkl", "rb").read(),
                            file_name=f"{name_model}.pkl"
                        )
                else:
                    pn = st.pills("Proceed to download Model or not",[True, False])
                    if pn:
                        st.session_state.save_create_clus = save_model(st.session_state.create_clus, "normal model")
                        st.success(f"Model saved as 'normal model'.")
                        st.download_button(
                            label="Download Model",
                            data=open(f"{name_model}.pkl", "rb").read(),
                            file_name=f"{name_model}.pkl"
                        )

            else:
                st.warning("No model available to save.")
        elif choose_model_to_save == "Ensembled Model":
            if st.session_state.ensemble_clus is not None:
                if st.toggle("Provide a custom name for the ensemble model"):
                    name_ensemble = st.text_input("Enter ensemble model name")
                    if name_ensemble:
                        st.session_state.save_ensemble_clus = save_model(st.session_state.ensemble_clus, name_ensemble)
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
                        st.session_state.save_ensemble_clus = save_model(st.session_state.ensemble_clus, name_ensemble)
                        st.success(f"Ensemble model saved as {name_ensemble}.")
                        st.download_button(
                            label="Download Ensemble Model",
                            data=open(f"{name_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_ensemble}.pkl"
                        )
            else:
                st.warning("No ensemble model available to save.")
        elif choose_model_to_save == "Tuned Normal Model":
            if st.session_state.tune_clus is not None:
                if st.toggle("Provide a custom name for the tuned model"):
                    name_tune = st.text_input("Enter tuned model name")
                    if name_tune:
                        st.session_state.save_tune_clus = save_model(st.session_state.tune_clus, name_tune)
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
                        st.session_state.save_tune_clus = save_model(st.session_state.tune_clus, name_tune)
                        st.success(f"Tuned model saved as {name_tune}.")
                        st.download_button(
                            label="Download Tuned Model",
                            data=open(f"{name_tune}.pkl", "rb").read(),
                            file_name=f"{name_tune}.pkl"
                        )
                        
            else:
                st.warning("No tuned model available to save.")
        elif choose_model_to_save == "Tuned Ensembled Model":
            if st.session_state.tune_ensembled_clus is not None:
                if st.toggle("Provide a custom name for the tuned ensemble model"):
                    name_tune_ensemble = st.text_input("Enter tuned ensemble model name")
                    if name_tune_ensemble:
                        st.session_state.save_ensembled_tune_clus = save_model(st.session_state.tune_ensembled_clus, name_tune_ensemble)
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
                        st.session_state.save_ensembled_tune_clus = save_model(st.session_state.tune_ensembled_clus, name_tune_ensemble)
                        st.success(f"Tuned ensemble model saved as {name_tune_ensemble}.")
                        st.download_button(
                            label="Download Tuned Ensemble Model",
                            data=open(f"{name_tune_ensemble}.pkl", "rb").read(),
                            file_name=f"{name_tune_ensemble}.pkl"
                        )             
            else:
                st.warning("No tuned ensemble model available to save.")

                    
        
    def reCreate(self):
        st.session_state.create_clus = None
        st.session_state.create_clus_df = None
        st.session_state.predict_clus = None
        st.session_state.predict_clus_df = None
        st.session_state.ensemble_clus = None
        st.session_state.ensemble_clus_df = None
        st.session_state.tune_clus = None
        st.session_state.tune_clus_df = None
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
