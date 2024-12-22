import streamlit as st
import pandas as pd
from pycaret.classification import *

classificationList = ["setup_cls","create_cls_df", "create_cls","predict_cls","predict_cls_df", "ensemble_cls","ensemble_cls_df", "tune_cls","tune_cls_df", "save_create_cls", "save_enseble_cls", "save_tune_cls"]
for i in classificationList:
    if i not in st.session_state:
        st.session_state[i] = None
class Classification:
    def __init__(self, dataset):
        self.dataset = dataset
        self.list = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']

    def selectTarget(self):
        if st.session_state.setup_cls is None:
            st.subheader("First setup your environment")
            st.divider()
            target = st.pills("Select the target column", self.dataset.columns)
            if target:
                st.session_state.setup_cls = setup(data=self.dataset, target=target)
                st.session_state.setup_cls = pull()
                st.dataframe(st.session_state.setup_cls)
        else:
            st.subheader("Target column is already selected")
            st.dataframe(st.session_state.setup_cls)

    def compareModels(self):
        if st.session_state.compare_cls is None and st.session_state.setup_cls is not None:
            st.session_state.compare_cls = compare_models()
            st.session_state.compare_cls = pull()
            st.dataframe(st.session_state.compare_cls)
        else:
            st.markdown("Comparison of models is already done")
            st.dataframe(st.session_state.compare_cls)

        if st.session_state.setup_cls is None:
            st.markdown("First select Target-->Compare models --> Create models then next steps")

    def createModel(self):
        self.selectTarget()
        self.compareModels()
        if st.session_state.create_cls is None and st.session_state.setup_cls is not None:
            choosedModel = st.pills("Select the desired model that you want to assess", self.list)
            if choosedModel:
                st.session_state.create_cls = create_model(choosedModel)
                st.session_state.create_cls_df = pull()
                st.dataframe(st.session_state.create_cls_df)
                st.session_state.predict_cls = predict_model(st.session_state.create_cls)
                st.session_state.predict_cls_df = pull()
                st.dataframe(st.session_state.predict_cls_df)
        else:
            st.info("Model is already created")
            st.dataframe(st.session_state.create_cls_df)
            st.dataframe(st.session_state.predict_cls_df)

    def ensembleModel(self):
        if st.session_state.ensemble_cls is None and st.session_state.create_cls is not None:
            st.session_state.ensemble_cls = ensemble_model(st.session_state.create_cls)
            st.session_state.ensemble_cls_df = pull()
            st.dataframe(st.session_state.ensemble_cls_df)
        else:
            st.info("Ensemble model is already created or not applicable")
            if st.session_state.ensemble_cls is not None:
                st.dataframe(st.session_state.ensemble_cls_df)

    def tuneModel(self):
        if st.session_state.tune_cls is None and st.session_state.create_cls is not None:
            st.session_state.tune_cls = tune_model(st.session_state.create_cls)
            st.session_state.tune_cls_df = pull()
            st.dataframe(st.session_state.tune_cls_df)
        else:
            st.info("Model is already tuned or not applicable")
            if st.session_state.tune_cls is not None:
                st.dataframe(st.session_state.tune_cls_df)
    def runApp(self):
        col1, col2 = st.columns([1, 2])

        with col1:
            options = st.pills("Select the stage", ["Create model", "Ensemble Model", "Tune Model"])
        with col2:
            classification = Classification(dataset=pd.DataFrame())  # Replace with your actual dataset
            if options == "Create model":
                classification.createModel()
            if options == "Ensemble Model":
                classification.ensembleModel()
            if options == "Tune Model":
                classification.tuneModel()
