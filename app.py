import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------
# Caching model & tokenizer
# ------------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("next_word_lstm.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

# Reverse word index for fast lookup
index_word = {v: k for k, v in tokenizer.word_index.items()}

max_sequence_len = model.input_shape[1] + 1

# ------------------------------------
# Prediction logic
# ------------------------------------
def predict_next_words(model, tokenizer, text, top_k=5, temperature=1.0):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    token_list = token_list[-(max_sequence_len - 1):]
    padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    preds = model.predict(padded, verbose=0)[0]

    # Temperature scaling
    preds = np.log(preds + 1e-9) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))

    top_indices = np.argsort(preds)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        word = index_word.get(idx, "<UNK>")
        confidence = preds[idx]
        results.append((word, confidence))

    return results

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.title("🧠 Next Word Prediction with LSTM")

input_text = st.text_input(
    "Enter text",
    value="To be or not to",
    help="Type a sentence and get next-word predictions"
)

col1, col2 = st.columns(2)
top_k = col1.slider("Top-K Predictions", 1, 10, 5)
temperature = col2.slider("Temperature", 0.2, 1.5, 1.0)

if input_text.strip():
    predictions = predict_next_words(
        model,
        tokenizer,
        input_text,
        top_k=top_k,
        temperature=temperature
    )

    if predictions:
        st.subheader("🔮 Predictions")
        for word, conf in predictions:
            st.write(f"**{word}** — {conf:.2%}")
    else:
        st.warning("Not enough context to predict the next word.")

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("Built with LSTM • Streamlit • TensorFlow")
