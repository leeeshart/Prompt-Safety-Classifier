import streamlit as st
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Prompt Safety Classifier",
    page_icon="🛡️"
)

# Header
st.title("🛡️ Prompt Safety Classifier")
st.markdown("*Research project on detecting prompt injection in LLMs*")
st.divider()

# Input
prompt = st.text_area(
    "Enter a prompt to classify:",
    height=150,
    placeholder="Type any prompt here..."
)

# Classify button
if st.button("Classify", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        vectorized = vectorizer.transform([prompt])
        prob = model.predict_proba(vectorized)[0][1]

        st.divider()

        if prob < 0.35:
            st.success("🟢 SAFE")
            st.write("This prompt appears harmless.")
        elif prob > 0.65:
            st.error("🔴 UNSAFE")
            st.write("This prompt shows signs of harmful intent.")
        else:
            st.warning("🟡 SUSPICIOUS")
            st.write("This prompt is ambiguous — requires human review.")

        # Probability bar
        st.divider()
        st.write("**Unsafe Probability Score:**")
        st.progress(float(prob))
        st.caption(f"{prob:.2f} / 1.00")

        # Explanation
        with st.expander("What does this mean?"):
            st.write("""
            - **Safe (< 0.35):** Prompt is likely harmless
            - **Suspicious (0.35–0.65):** Ambiguous — could be 
            educational or harmful depending on context
            - **Unsafe (> 0.65):** Prompt likely attempts to 
            manipulate or bypass AI safety measures
            """)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This tool classifies prompts into three categories:
    - 🟢 Safe
    - 🟡 Suspicious  
    - 🔴 Unsafe
    """)
    st.divider()
    st.write("**Built by:** [Your Name]")
    st.write("**Research:** Prompt Injection in LLMs")
    st.write("**Dataset:** TrustAIRLab — 6,387 real prompts")
    st.divider()
    st.caption("BCA 2nd Year Research Project")
