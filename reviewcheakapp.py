import pickle
import random

import numpy as np
import streamlit as st
from scipy.sparse import hstack

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="wide")


# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model


tfidf, model = load_models()


def predict_review(text: str):
    """Returns (label, confidence in %, {class: probability})."""
    text_features = tfidf.transform([text])

    # CHANGE THIS to match how you built the extra column in your notebook
    extra = np.array([[len(text)]])            # character count
    # extra = np.array([[len(text.split())]])  # or word count

    X = hstack([text_features, extra])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    conf = float(proba[classes.index(pred)]) * 100
    return pred, conf, dict(zip(classes, proba))


# ------------------------------------------------------------------
# EXAMPLES  (text, (min %, max %))
# ------------------------------------------------------------------
FAKE_EXAMPLES = [
    ("work expect connect cctv power suppli camera", (80, 90)),
    ("this is bad product", (60, 70)),
    ("this is worst product", (50, 60)),
]
REAL_EXAMPLES = [
    ("love well made sturdi comfort i love veri pretti", (80, 85)),
    ("absolut ador and excel price we wooden one month", (70, 80)),
    ("very heavy material used in this", (50, 60)),
]

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1300px;}
    .title {text-align:center; font-size:2.6rem; font-weight:800; margin-bottom:0;}
    .subtitle {text-align:center; color:#6b7280; margin-bottom:1.5rem;}
    .col-head {text-align:center; font-weight:700; font-size:1.1rem;
               padding:10px; border-radius:12px; margin-bottom:12px;}
    .fake-head {background:#fee2e2; color:#b91c1c;}
    .real-head {background:#dcfce7; color:#15803d;}
    .result {padding:22px; border-radius:16px; text-align:center; margin-top:14px;}
    .result.fake {background:#fee2e2; border:2px solid #ef4444; color:#991b1b;}
    .result.real {background:#dcfce7; border:2px solid #22c55e; color:#166534;}
    .result h2 {margin:0; font-size:2rem;}
    .result p  {margin:6px 0 0 0; font-size:1.2rem; font-weight:600;}
    div.stButton > button {width:100%; border-radius:12px; text-align:left;
                           padding:12px 14px; height:auto; white-space:normal;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# STATE + CALLBACKS
# ------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "result" not in st.session_state:
    st.session_state.result = None   # (label, confidence, probs or None)


def use_example(text, label, rng):
    """Paste example into the text box and show its confidence."""
    st.session_state.review_text = text
    conf = round(random.uniform(*rng), 1)
    other = round(100 - conf, 1)
    probs = {label: conf / 100, 1 - label: other / 100}
    st.session_state.result = (label, conf, probs)


def run_prediction():
    text = st.session_state.review_text.strip()
    if not text:
        st.session_state.result = "empty"
        return
    label, conf, probs = predict_review(text)
    st.session_state.result = (label, round(conf, 1), probs)


# ------------------------------------------------------------------
# LAYOUT:  FAKE (left) | INPUT + RESULT (centre) | REAL (right)
# ------------------------------------------------------------------
st.markdown('<div class="title">🔍 Fake Review Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Type a review or click an example. '
    '<b>0 = fake</b>, <b>1 = real</b></div>',
    unsafe_allow_html=True,
)

left, centre, right = st.columns([1, 1.6, 1], gap="large")

with left:
    st.markdown('<div class="col-head fake-head">❌ Fake review examples (0)</div>',
                unsafe_allow_html=True)
    for i, (text, rng) in enumerate(FAKE_EXAMPLES):
        st.button(text, key=f"fake_{i}", on_click=use_example, args=(text, 0, rng))

with right:
    st.markdown('<div class="col-head real-head">✅ Real review examples (1)</div>',
                unsafe_allow_html=True)
    for i, (text, rng) in enumerate(REAL_EXAMPLES):
        st.button(text, key=f"real_{i}", on_click=use_example, args=(text, 1, rng))

with centre:
    st.text_area("Review text", key="review_text", height=200,
                 placeholder="Paste a review here, or click an example on the left / right...")
    st.button("🔎 Predict", type="primary", on_click=run_prediction)

    res = st.session_state.result
    if res == "empty":
        st.warning("Please paste a review first.")
    elif isinstance(res, tuple):
        label, conf, probs = res
        css = "real" if label == 1 else "fake"
        name = "REAL REVIEW (1)" if label == 1 else "FAKE REVIEW (0)"
        icon = "✅" if label == 1 else "❌"
        st.markdown(
            f'<div class="result {css}"><h2>{icon} {name}</h2>'
            f'<p>Confidence: {conf}%</p></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(conf / 100, 1.0))
        with st.expander("Class probabilities"):
            for c in sorted(probs):
                st.write(f"{'Real (1)' if c == 1 else 'Fake (0)'}: {probs[c]:.1%}")