import streamlit as st
import pickle
import numpy as np



with open("models/vectorizer_company_profile.pkl", "rb") as f:
  vectorizer_company_profile = pickle.load(f)

with open("models/vectorizer_description.pkl", "rb") as f:
  vectorizer_description = pickle.load(f)

with open("models/vectorizer_requirements.pkl", "rb") as f:
  vectorizer_requirements = pickle.load(f)

with open("models/ANN_model.pkl", "rb") as f:
  ANN_model = pickle.load(f)


st.title("Fake Job Prediction")
st.subheader("Created by Purab Banerjee")

company_profile = st.text_input("Company Profile")
description = st.text_input("Job Description")
requirements = st.text_input("Requirements")

submit = st.button("Predict")

# def clean_text(text):
#   return text.str.replace("[^a-zA-Z0-9 ]", "", regex=True)

if submit:
  

  cleaned_company_profile = company_profile
  cleaned_description = description
  cleaned_requirements = requirements

  tfidf_company_profile = vectorizer_company_profile.transform([cleaned_company_profile])
  tfidf_company_description = vectorizer_description.transform([cleaned_description])
  tfidf_company_requirements = vectorizer_requirements.transform([cleaned_requirements])

  tfidf_data = np.hstack((tfidf_company_profile.toarray(), tfidf_company_description.toarray(), tfidf_company_requirements.toarray()))



  prediction_dt = ANN_model.predict(tfidf_data)[0]

  if prediction_dt == 0:
    st.write("**Prediction:** Not Fake")
  else:
    st.write("**Prediction:** Fake")

  # You can use the predictions from both models here for further analysis or combine them using appropriate logic.
    

