import os
import google.generativeai as genai
from rouge import Rouge

def load_gemini():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.0-flash")

GEMINI = load_gemini()

def evaluate_with_llm(question, answer, context):
    prompt = f"""
Evaluate the RAG answer.

Question: {question}
Context: {context}
Answer: {answer}

Respond as:
Score: <0-10>
Reason: <one sentence>
"""
    response = GEMINI.generate_content(prompt)
    return response.text

def evaluate_with_rouge(answer, context):
    rouge = Rouge()
    return rouge.get_scores(answer, context)[0]

if __name__ == "__main__":
    print(evaluate_with_llm("What is AI?", "AI is...", "AI is..."))
