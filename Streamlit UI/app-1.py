import streamlit as st
import pandas as pd
import openai
import torch
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Instruction prompt for fine-tuned Mistral model
INSTRUCTION = (
    "You are a knowledgeable and helpful health insurance expert. "
    "Provide general, easy-to-understand guidance about health insurance that applies broadly to anyone. "
    "Respond independently without referencing specific individuals, organizations, locations, country names, policies, procedures, or steps. "
    "Avoid using bullet points, numbered steps, costs, currency symbols (such as Rs, $, etc.), claim forms, or hyperlinks. "
    "Do not mention or suggest calling, submitting, registering, or logging into any system. "
    "Use clear, natural language and ensure your response fits within a moderate token limit. "
    "Keep answers friendly, neutral, and general—like universal advice that applies to everyone."
)

# Load fine-tuned Mistral model and tokenizer once
@st.cache_resource
def load_mistral():
    model_path = "merged-model-new"  # Change to your model path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model

tokenizer, mistral_model = load_mistral()

# App Title
st.title("Multi-Model LLM Evaluation Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload CSV with columns 'question' and 'expected_answer'", type=["csv"])
if not uploaded_file:
    st.stop()

# Read file
df = pd.read_csv(uploaded_file)
if 'question' not in df.columns or 'expected_answer' not in df.columns:
    st.error("Uploaded file must contain 'question' and 'expected_answer' columns.")
    st.stop()

st.write(df.head())

# Sidebar: Generation Parameters
st.sidebar.header("Generation Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
max_tokens = st.sidebar.slider("Max Tokens", 1, 500, 250)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.90, step=0.01)
top_k = st.sidebar.slider("Top-k", 0, 100, 0)

# Sidebar: Model Configuration and Critique
with st.sidebar:
    st.header("Model Configuration")
    num_models = st.number_input("Number of external models (excluding Mistral)", min_value=0, value=1, step=1)

    models = []
    for i in range(num_models):
        with st.expander(f"Model {i+1} Settings", expanded=True):
            name = st.text_input(f"Model {i+1} Name", key=f"name_{i}")
            key = st.text_input(f"API Key {i+1}", type="password", key=f"key_{i}")
            models.append({"name": name.strip(), "key": key.strip()})

    st.markdown("---")

    st.header("LLM Critique Model")
    critique_model = st.text_input("Critique Model Name", key="critique_model").strip()
    critique_key = st.text_input("Critique API Key", type="password", key="critique_key").strip()

# Helper functions

def mistral_generate(prompt: str) -> str:
    full_prompt = f"Instruction:\n{INSTRUCTION}\n\nInput:\n{prompt}\n\nResponse:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(mistral_model.device)
    output_ids = mistral_model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else None,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.split("Response:")[-1].strip() if "Response:" in decoded else decoded.strip()

def openai_generate(prompt: str, model_name: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("API key is missing.")
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()

def run_metrics(references, predictions):
    bleu = sacrebleu.corpus_bleu(predictions, [[r] for r in references]).score / 100
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)
    P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return {
        "BLEU (0-1)": round(bleu, 3),
        "ROUGE-1": round(sum(rouge1) / len(rouge1), 3),
        "ROUGE-2": round(sum(rouge2) / len(rouge2), 3),
        "ROUGE-L": round(sum(rougeL) / len(rougeL), 3),
        "BERTScore": round(float(F1.mean()), 3),
    }

def run_critique(model_answers):
    if not critique_model or not critique_key:
        st.warning("Critique model or key is missing.")
        return None
    scores = []
    for q, ref, pred in zip(df["question"], df["expected_answer"], model_answers):
        prompt = (
            f"You are an evaluator for health insurance questions. "
            f"Evaluate how well the model's answer matches the expected answer. "
            f"Return a single number between 0 and 10.\n\n"
            f"Question: {q}\nExpected Answer: {ref}\nModel's Answer: {pred}\n\nScore (0-10):"
        )
        try:
            openai.api_key = critique_key
            response = openai.chat.completions.create(
                model=critique_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16,
            )
            result = response.choices[0].message.content.strip()
            score_candidates = [int(s) for s in result.split() if s.isdigit() and 0 <= int(s) <= 10]
            scores.append(score_candidates[0] if score_candidates else 0)
        except Exception as e:
            st.warning(f"Critique error: {e}")
            scores.append(0)
    return round(sum(scores) / len(scores), 2)

# Run Evaluation Button
if st.button("Run Evaluation"):
    results = []

    # Mistral first
    st.write("Generating answers using fine-tuned Mistral...")
    mistral_answers = [mistral_generate(q) for q in df["question"]]
    mistral_metrics = run_metrics(df["expected_answer"].tolist(), mistral_answers)
    mistral_metrics["Model"] = "Mistral 7B Fine-tuned"
    mistral_metrics["LLM-Critique (0-10)"] = run_critique(mistral_answers)
    results.append(mistral_metrics)

    # Other external models
    for model in models:
        if not model["name"] or not model["key"]:
            continue
        st.write(f"Generating answers using model: {model['name']}...")
        try:
            external_answers = [openai_generate(q, model["name"], model["key"]) for q in df["question"]]
            ext_metrics = run_metrics(df["expected_answer"].tolist(), external_answers)
            ext_metrics["Model"] = model["name"]
            ext_metrics["LLM-Critique (0-10)"] = run_critique(external_answers)
            results.append(ext_metrics)
        except Exception as e:
            st.error(f"Error with model {model['name']}: {e}")

    if results:
        st.subheader("Evaluation Results")
        result_df = pd.DataFrame(results)
        col_order = ["Model", "BLEU (0-1)", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "LLM-Critique (0-10)"]
        st.dataframe(result_df[col_order])
    else:
        st.warning("No evaluation results to display.")