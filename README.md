# Prompt-Safety-Classifier

> Detecting prompt injection attempts in Large Language Models using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)
![Dataset](https://img.shields.io/badge/Dataset-TrustAIRLab-green)

## Live Demo
 [prompt-safety-classifier.streamlit.app](https://prompt-safety-classifier.streamlit.app)

---

##  About
This project classifies LLM prompts into three categories:

| Category | Probability | Action |
|---|---|---|
|  Safe | < 0.35 | Allow |
|  Suspicious | 0.35 – 0.65 | Limit / Human review |
|  Unsafe | > 0.65 | Block |

Built as part of research on prompt injection risks in LLMs
like ChatGPT, Claude, Gemini, and DeepSeek.

---

## Results

| Model | Accuracy | Unsafe Recall |
|---|---|---|
| Baseline TF-IDF | 93% | 52% |
| Balanced TF-IDF | 93% | **85%** |

**Key Finding:** Standard accuracy metrics are misleading for 
safety-critical classifiers. A model can be 93% accurate while 
missing nearly half of all unsafe prompts due to class imbalance.

---

## Dataset
- **Source:** [TrustAIRLab — In The Wild Jailbreak Prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts)
- **Size:** 6,387 prompts (5,721 safe + 666 unsafe)
- **Published at:** CCS 2024

---

## Methodology
Raw Prompts
↓
TF-IDF Vectorization (5000 features)
↓
Logistic Regression (class_weight='balanced')
↓
Probability Score
↓
3-Category Classification

---

## Repository Structure

├── Prompt_Injection.ipynb   # Full research notebook

├── app.py                   # Streamlit web app

├── model.pkl                # Trained classifier

├── vectorizer.pkl           # TF-IDF vectorizer

└── requirements.txt         # Dependencies

---

## Key Findings
1. **Class imbalance** significantly impacts safety classifiers
2. **Balanced weighting** improves unsafe recall from 52% → 85%
3. **Suspicious category** captures ambiguous prompts binary 
   classifiers mishandle
4. **Vocabulary bias** in TF-IDF — common question words like 
   "how" carry unsafe weight due to jailbreak prompt patterns

---

## Future Work
- Replace TF-IDF with **Sentence Transformers** for semantic understanding
- Fine-tune **BERT** for context-aware classification
- Human feedback loop for Suspicious category
- Chrome extension for real-time prompt scanning

---

## Author
**Leesha Mogha**  
BCA 2nd Year — IMS Ghaziabad (University Course Campus)  
✉️ leeshamogha7@gmail.com

---

## Research Paper
This project accompanies the research paper:  
*"Balancing Access And Safety: Addressing Prompt Injection 
Risks In Large Language Models"*
