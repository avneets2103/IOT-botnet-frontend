import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

nav = st.sidebar.selectbox("What do you wanna do? ", options=['Use the model'])
if nav == 'Use the model':
    nav2 = st.sidebar.selectbox("Predict on:", options=['Sample data', 'Your own data'])
    with open('SVMDanger.pkl', 'rb') as file:
        model = pickle.load(file)
        if(nav2=='Sample data'):
            sz = st.number_input("Enter sample size", min_value=32, max_value=1000, step=10)
            sampleData = pd.read_csv('dangerDF.csv')
            sampleData.drop('Unnamed: 0', axis=1, inplace=True)
            sampleD = sampleData.sample(sz)
            st.write(sampleD)
            st.write(sampleD.shape)
            if st.checkbox("Predict on this sample"):
                X = sampleData
                import joblib
                pca = joblib.load('pca.joblib')
                X = pca.fit_transform(StandardScaler().fit_transform(sampleD.drop(['danger'], axis=1)))
                ypreds = pd.DataFrame(model.predict(X), columns=['output'])
                y = sampleD.danger
                y = y.reset_index(drop=True)
                show = pd.concat([y, ypreds], axis=1)
                st.write(pd.DataFrame(show))
        elif nav2=='Your own data':
            upload = st.file_uploader("Enter your csv file with 115 features as mentioned in the about section")
            if upload is None:
                st.write("Upload again")
            else:
                inputCSV = pd.read_csv(upload)
                st.write(inputCSV)
                if st.checkbox("Predict on this sample"):
                    import joblib
                    pca = joblib.load('pca.joblib')
                    X = pca.fit_transform(StandardScaler().fit_transform(inputCSV))
                    ypreds = pd.DataFrame(model.predict(X), columns=['output'])
                    st.write(pd.DataFrame(ypreds))


