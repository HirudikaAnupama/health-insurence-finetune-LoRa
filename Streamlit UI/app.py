import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import re

st.set_page_config(page_title="Health Insurance Model", layout="wide")
st.title("Health Insurance Assistant - Fine-Tuned Mistral")

@st.cache_resource
def load_model():
    base_model = "mistralai/Mistral-7B-v0.1"
    lora_path = "mistral-lora-attempt-new-10"

    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

instruction = (
    "You are a knowledgeable and helpful health insurance expert. "
    "Provide general, easy-to-understand guidance about health insurance that applies broadly to anyone. "
    "Respond independently without referencing specific individuals, organizations, locations, country names, policies, procedures, or steps. "
    "Avoid using bullet points, numbered steps, costs, currency symbols, claim forms, or hyperlinks. "
    "Do not mention or suggest calling, submitting, registering, or logging into any system. "
    "Use clear, natural language and ensure your response fits within a moderate token limit. "
    "Keep answers friendly, neutral, and general like universal advice that applies to everyone."
)

def build_prompt(instruction, question):
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{question}\n\n"
        f"### Response:\n"
    )

def truncate_to_complete_sentence(text):
    match = re.search(r'^(.*?[\.!?])[^\.!?]*$', text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# Sidebar for generation settings
st.sidebar.header("Generation Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 250, step=10)

# User input
user_question = st.text_area("Enter your health insurance question", placeholder="Example: Why do we need health insurance?")

if st.button("Generate Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            prompt = build_prompt(instruction, user_question)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

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
            answer = truncate_to_complete_sentence(raw_answer)

        st.subheader("Model Response")
        st.write(answer)
