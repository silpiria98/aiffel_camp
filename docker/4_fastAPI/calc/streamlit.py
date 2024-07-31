import streamlit as st
import requests
import json

st.title("Ther calculator")
st.subheader("prototype")

x = st.text_area("x:")
operator = st.selectbox("연산자", ("+", "-", "*", "/"))
y = st.text_area("y")

info_dict = {"x": x, "y": y, "operator": operator}

if st.button("Calculator"):
    result = requests.post(
        "http:127.0.0.1:8000/calculator", data=json.dumps(info_dict), verify=False
    )
    st.subheader(f"result: {result.text}")
