import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re
import time

st.set_page_config(page_title="Health Insurance Assistant", layout="wide")
st.title("Health Insurance Assistant")

@st.cache_resource
def load_model():
    model_path = "merged-model-new"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_classifier():
    classifier_path = "./distilbert-health-insurance-classifier"
    clf_tokenizer = DistilBertTokenizer.from_pretrained(classifier_path)
    clf_model = DistilBertForSequenceClassification.from_pretrained(classifier_path)
    clf_model.eval()
    return clf_tokenizer, clf_model

tokenizer, model = load_model()
clf_tokenizer, clf_model = load_classifier()

instruction = (
    "You are a knowledgeable and helpful health insurance expert. "
    "Only answer questions that are clearly related to health insurance. "
    "If the question is unrelated to health insurance, politely respond by saying you can only assist with health insurance topics. "
    "When answering, provide general, easy-to-understand guidance that applies broadly to anyone. "
    "Do not reference specific individuals, organizations, locations, country names, policies, procedures, or steps. "
    "Avoid using bullet points, numbered steps, costs, currency symbols, claim forms, or hyperlinks. "
    "Do not mention or suggest calling, submitting, registering, or logging into any system. "
    "Use clear, natural language and ensure your response fits within a moderate token limit. "
    "Keep answers friendly, neutral, and general like universal advice that applies to everyone."
)

def build_prompt(instruction, question):
    return f"### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:\n"

def truncate_to_complete_sentence(text):
    match = re.search(r'^(.*?[\.!?])[^\.!?]*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def is_health_insurance_related(text):
    inputs = clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class == 1  # 1 = Health Insurance

# Sidebar settings
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 250, step=10)

# Text input
user_question = st.text_area("Ask your question related to health insurance:", placeholder="Example: How do I file an insurance claim?")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    elif is_health_insurance_related(user_question):
        prompt = build_prompt(instruction, user_question)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with st.spinner("Generating response..."):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    max_new_tokens=max_tokens,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        raw_answer = decoded.split("### Response:\n")[-1].strip()
        final_answer = truncate_to_complete_sentence(raw_answer)

        # Typing animation
        st.subheader("Response:")
        words = final_answer.split()
        output_placeholder = st.empty()
        animated_text = ""

        for word in words:
            animated_text += word + " "
            output_placeholder.markdown(f"<p style='font-size:20px'>{animated_text}</p>", unsafe_allow_html=True)
            time.sleep(0.08)  # Adjust speed here (seconds per word)

    else:
        st.subheader("Response:")
        st.info("I'm here to assist with health insurance questions only. Please ask something related to health insurance.")
