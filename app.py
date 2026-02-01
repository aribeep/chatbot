import streamlit as st
from PIL import Image
import io
import base64
import os
import tempfile
import json
from intent import predict_intent
from sentiment import predict_sentiment
from aircraft import predict_aircraft
from aircraft2 import predict_aircraft_variant
from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "gradcam_outputs")
os.makedirs(GRADCAM_DIR, exist_ok=True)
@app.route("/gradcam/<filename>")
def gradcam_file(filename):
    return send_from_directory(GRADCAM_DIR, filename)
with open('word2idx.json', 'r') as file:
        word2idx = json.load(file)

# Show title and description.
st.title("Changi Virtual Assistant")
st.write(
    "This chatbot accepts both text and image inputs. "
    "Upload an image or type your message to start chatting!"
)

# Create a session state variable to store the chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["Intent & Sentiment", "Image Classification"])

with tab1:
    st.header("Intent & Sentiment Analysis")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])

    # Chat input for text messages
    if prompt := st.chat_input("Type your query here..."):
        # Create message object
        message_obj = {
            "role": "user", 
            "content": prompt,
            "type": "text"
        }
        
        # Store and display the user's message
        st.session_state.messages.append(message_obj)
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            response = ""
            response = "The intent and sentiment is",predict_intent(prompt), predict_sentiment(prompt, word2idx)
            st.markdown(response)
        
        # Store the assistant's response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "type": "text"
        })

with tab2:
    st.header("Image Classification")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        help="Upload an image to analyze or discuss"
    )

    

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image")

        # Save to a temp file so predict_aircraft(path) works
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            suffix = ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        st.success("Image uploaded successfully!")

        with st.chat_message("user"):
            st.markdown("Uploaded an image.")

        with st.chat_message("assistant"):
            family_label, family_conf = predict_aircraft(temp_path)
            variant_result = predict_aircraft_variant(
            temp_path,
                checkpoint_path="./aircraft_varient.pt",
                variants_txt_path="variants.txt"
        )           

        response = (
    f"**Aircraft family:** {family_label}\n"
    f"**Family confidence:** {family_conf:.2%}\n\n"
    f"**Predicted variant:** {variant_result['pred_top1']}\n\n"
    f"**Top-3 variants:**\n"
        )

        for v in variant_result["top3"]:
            response += f"- {v['label']} ({v['confidence']:.2%})\n"
                # Store the assistant's response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "type": "text"
        })

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass


# Add a clear chat button in the sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()