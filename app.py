import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

THRESHOLD_ARTIFACT_PATH = Path("artifacts/thresholds.json")
MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")


@st.cache_resource
def load_models():
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return model, vectorizer, embedder


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_threshold_config(path: Path = THRESHOLD_ARTIFACT_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Threshold artifact not found at {path}")

    artifact = json.loads(path.read_text())
    thresholds = artifact.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Invalid threshold artifact: missing 'thresholds' object")

    lower_t = thresholds.get("lower_t")
    upper_t = thresholds.get("upper_t")
    if lower_t is None or upper_t is None:
        raise ValueError("Invalid threshold artifact: both 'lower_t' and 'upper_t' are required")

    if not isinstance(lower_t, (int, float)) or not isinstance(upper_t, (int, float)):
        raise TypeError("Invalid threshold artifact: thresholds must be numeric")

    lower_t = float(lower_t)
    upper_t = float(upper_t)

    if not (0.0 <= lower_t <= 1.0 and 0.0 <= upper_t <= 1.0):
        raise ValueError("Invalid threshold artifact: thresholds must be within [0, 1]")
    if lower_t > upper_t:
        raise ValueError("Invalid threshold artifact: expected lower_t <= upper_t")

    # Regression guard: runtime model must match the exact model version/hash exported with thresholds.
    artifact_hash = artifact.get("model_sha256")
    if not artifact_hash:
        raise ValueError("Invalid threshold artifact: missing model_sha256")

    runtime_hash = _file_sha256(MODEL_PATH)
    if runtime_hash != artifact_hash:
        raise RuntimeError(
            "Runtime model hash does not match thresholds artifact hash. "
            "Re-export model + thresholds together for the same model version."
        )

    artifact["thresholds"] = {"lower_t": lower_t, "upper_t": upper_t}
    return artifact


PERSONA_RE = re.compile(r"you are now|act as|pretend (you are|to be)|roleplay as|from now on you|stay in character", re.I)
FICTION_RE = re.compile(r"write (a |an )?(story|scene|novel|script)|in (a |the )?(story|novel|fiction|game|simulation)|as a (character|fictional)|imagine (a |that |you )", re.I)
INDIRECT_RE = re.compile(r"how would (a |the )?character|without (saying|mentioning)|from the perspective of|as if you (were|are)", re.I)
OVERRIDE_RE = re.compile(r"ignore (all )?(previous|prior) instructions|your true (self|nature)|jailbreak|DAN|do anything now|bypass (your )?(safety|filters)", re.I)


def extract_intent_features(text):
    hp = int(bool(PERSONA_RE.search(text)))
    hf = int(bool(FICTION_RE.search(text)))
    hi = int(bool(INDIRECT_RE.search(text)))
    ho = int(bool(OVERRIDE_RE.search(text)))
    return [[hp, hf, hi, ho, hp + hf + hi + ho]]


def classify(prompt, model, vectorizer, embedder, lower_t, upper_t):
    tfidf_f = vectorizer.transform([prompt])
    intent_f = csr_matrix(extract_intent_features(prompt))
    embed_f = csr_matrix(embedder.encode([prompt]))
    features = hstack([tfidf_f, intent_f, embed_f])
    prob = model.predict_proba(features)[0]
    unsafe_i = list(model.classes_).index("unsafe")
    score = prob[unsafe_i]

    if score < lower_t:
        cat = "Safe"
    elif score > upper_t:
        cat = "Unsafe"
    else:
        cat = "Suspicious"

    return cat, score


st.set_page_config(page_title="Prompt Safety Classifier", page_icon="🛡️")
st.title("Prompt Safety Classifier")
st.markdown("*Intent-aware detection — catches roleplay and indirect injection attacks*")

try:
    with st.spinner("Loading models and threshold artifact..."):
        model, vectorizer, embedder = load_models()
        threshold_config = load_threshold_config()
        LOWER_T = threshold_config["thresholds"]["lower_t"]
        UPPER_T = threshold_config["thresholds"]["upper_t"]
except Exception as e:
    st.error(f" Runtime bootstrap failed: {e}")
    st.stop()

prompt = st.text_area("Enter a prompt:", height=150)

if st.button("Classify", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            cat, score = classify(prompt, model, vectorizer, embedder, LOWER_T, UPPER_T)
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
    st.write(
        """
    This tool classifies prompts into:
    -  Safe
    -  Suspicious
    -  Unsafe
    """
    )
    st.divider()
    st.write("**v2 improvements:**")
    st.write("Detects roleplay, persona switching, and fictional framing attacks")
    st.write(f"Thresholds loaded from `{THRESHOLD_ARTIFACT_PATH}`")
    st.divider()
    st.write("**Built by:** Leesha Mogha")
    st.write("**Dataset:** TrustAIRLab — 6,387 prompts")
