import streamlit as st
from rag_agent import ask 
from evaluation import evaluate_with_llm, evaluate_with_rouge

st.set_page_config(page_title="AI RAG Agent", page_icon="🤖", layout="centered")

st.title("🤖 AI RAG Agent (Gemini 2.0 + LangGraph + Chroma)")
st.write("Ask a question based on the `.txt` files in your **data/** folder.")

# User Input Box
question = st.text_input(
    "🔍 Enter your question:",
    placeholder="e.g., What is deep learning? Why is AI important?"
)

# Run button
if st.button("Run RAG Agent"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Thinking... Generating answer..."):
            response = ask(question)

        st.subheader("📘 Retrieved Context")
        st.write(response["context"] if response["context"] else "No context retrieved.")

        st.subheader("🤖 AI Answer")
        st.success(response["answer"])

        st.subheader("🔎 Reflection (Quality Check)")
        st.info(response["reflection"])

        st.subheader("🧠 LLM-as-a-Judge")
        judge_score = evaluate_with_llm(question, response["answer"], response["context"])
        st.write(judge_score)

        st.subheader("📈 ROUGE Score")
        rouge = evaluate_with_rouge(response["answer"], response["context"])
        st.write(rouge)