import streamlit as st
from PIL import Image
import io
import base64
from intent import model

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
            response += model.eval()
            st.markdown(response)
        
        # Store the assistant's response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "type": "text"
        })

with tab2:
    st.header("Image Classification")
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "gif", "bmp"],
        help="Upload an image to analyze or discuss"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=True)
        
        # Convert image to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format=image.format if hasattr(image, 'format') else "PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Store image in session state
        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = img_str
            st.session_state.image_format = image.format if hasattr(image, 'format') else "PNG"
        
        st.success("Image uploaded successfully!, classify image")
    
    # Clear image button
    if "uploaded_image" in st.session_state:
        if st.button("Clear Image"):
            del st.session_state.uploaded_image
            if "image_format" in st.session_state:
                del st.session_state.image_format
            st.rerun()


# Add a clear chat button in the sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()