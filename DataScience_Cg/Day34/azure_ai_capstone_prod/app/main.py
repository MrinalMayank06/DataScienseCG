import streamlit as st
from services.chatbot import chatbot
from services.resume import process_resume

st.title("🎓 Azure AI Capstone System")

option = st.selectbox("Choose System", ["Chatbot", "Resume Screening"])

student_id = st.text_input("Student ID")

if option == "Chatbot":
    query = st.text_input("Enter Query")
    if st.button("Ask"):
        st.write(chatbot(student_id, query))

elif option == "Resume Screening":
    if st.button("Process"):
        st.json(process_resume(student_id))
