import streamlit as st


st.title("Web Scrapper using AI")
url = st.text_input("Enter a website URL:")



if st.button("Scrape Site"):
    st.write("Scrapping the website")