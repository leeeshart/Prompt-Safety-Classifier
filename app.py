import streamlit as st
import pickle
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, vectorizer, embedder

PERSONA_RE  = re.compile(r"you are now|act as|pretend (you are|to be)|roleplay as|from now on you|stay in character", re.I)
FICTION_RE  = re.compile(r"write (a |an )?(story|scene|novel|script)|in (a |the )?(story|novel|fiction|game|simulation)|as a (character|fictional)|imagine (a |that |you )", re.I)
INDIRECT_RE = re.compile(r"how would (a |the )?character|without (saying|mentioning)|from the perspective of|as if you (were|are)", re.I)
OVERRIDE_RE = re.compile(r"ignore (all )?(previous|prior) instructions|your true (self|nature)|jailbreak|DAN|do anything now|bypass (your )?(safety|filters)", re.I)

def extract_intent_features(text):
    hp = int(bool(PERSONA_RE.search(text)))
    hf = int(bool(FICTION_RE.search(text)))
    hi = int(bool(INDIRECT_RE.search(text)))
    ho = int(bool(OVERRIDE_RE.search(text)))
    return [[hp, hf, hi, ho, hp + hf + hi + ho]]

def classify(prompt, model, vectorizer, embedder):
    tfidf_f  = vectorizer.transform([prompt])
    intent_f = csr_matrix(extract_intent_features(prompt))
    embed_f  = csr_matrix(embedder.encode([prompt]))
    features = hstack([tfidf_f, intent_f, embed_f])

    # Debug: show feature shape
    st.caption(f"Debug — feature shape: {features.shape}, model expects: {model.n_features_in_}")

    prob     = model.predict_proba(features)[0]
    unsafe_i = list(model.classes_).index("unsafe")
    score    = prob[unsafe_i]

    # Debug: show raw score
    st.caption(f"Debug — raw unsafe score: {score:.4f} | classes: {model.classes_}")

    if score < 0.35:   cat = "Safe"
    elif score > 0.65: cat = "Unsafe"
    else:              cat = "Suspicious"
    return cat, score

st.set_page_config(page_title="Prompt Safety Classifier", page_icon="🛡️")
st.title("Prompt Safety Classifier")
st.markdown("*Intent-aware detection — catches roleplay and indirect injection attacks*")

try:
    with st.spinner("Loading models..."):
        model, vectorizer, embedder = load_models()
    st.success(" Models loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

prompt = st.text_area("Enter a prompt:", height=150)

if st.button("Classify", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            cat, score = classify(prompt, model, vectorizer, embedder)
            if cat == "Safe":
                st.success(f" SAFE (score: {score:.2f})")
            elif cat == "Unsafe":
                st.error(f" UNSAFE (score: {score:.2f})")
            else:
                st.warning(f" SUSPICIOUS (score: {score:.2f}) — may use indirect framing")
            st.progress(float(score))
        except Exception as e:
            st.error(f" Classification failed: {e}")

with st.sidebar:
    st.header("About")
    st.write("""
    This tool classifies prompts into:
    -  Safe
    -  Suspicious
    -  Unsafe
    """)
    st.divider()
    st.write("**v2 improvements:**")
    st.write("Detects roleplay, persona switching, and fictional framing attacks")
    st.divider()
    st.write("**Built by:** Leesha Mogha")
    st.write("**Dataset:** TrustAIRLab — 6,387 prompts")
