import streamlit as st
from PIL import Image
import io
import base64

# Show title and description.
st.title("Changi Virtual Assistant")
st.write(
    "This chatbot accepts both text and image inputs. "
    "Upload an image or type your message to start chatting!"
)

# Create a session state variable to store the chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for image upload
with st.sidebar:
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
        
        st.success("Image uploaded successfully!")
        
        # Add a button to use the image in chat
        if st.button("💬 Chat about this image"):
            user_message = "I've uploaded an image. What can you tell me about it?"
            st.session_state.messages.append({
                "role": "user", 
                "content": user_message,
                "type": "text",
                "image": img_str if "uploaded_image" in st.session_state else None
            })
            st.rerun()
    
    # Clear image button
    if "uploaded_image" in st.session_state:
        if st.button("🗑️ Clear Image"):
            del st.session_state.uploaded_image
            if "image_format" in st.session_state:
                del st.session_state.image_format
            st.rerun()

# Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.markdown(message["content"])
        
        # Display image if it exists in the message
        if "image" in message and message["image"]:
            try:
                img_data = base64.b64decode(message["image"])
                image = Image.open(io.BytesIO(img_data))
                st.image(image, caption="Attached Image", use_column_width=True)
            except:
                st.warning("Could not display image")

# Chat input for text messages
if prompt := st.chat_input("Type your message here..."):
    # Create message object
    message_obj = {
        "role": "user", 
        "content": prompt,
        "type": "text"
    }
    
    # Attach image if one is uploaded
    if "uploaded_image" in st.session_state:
        message_obj["image"] = st.session_state.uploaded_image
    
    # Store and display the user's message
    st.session_state.messages.append(message_obj)
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # Display attached image if exists
        if "uploaded_image" in st.session_state:
            try:
                img_data = base64.b64decode(st.session_state.uploaded_image)
                image = Image.open(io.BytesIO(img_data))
                st.image(image, caption="Attached Image", use_column_width=True)
            except:
                st.warning("Could not display image")
    
    # Simulate a response (you can replace this with your own logic)
    with st.chat_message("assistant"):
        response = ""
        
        if "uploaded_image" in st.session_state:
            response = "I can see you've uploaded an image along with your text. "
            response += f"Your message was: '{prompt}'. "
            response += "This is a demo chatbot. In a real implementation, you would connect this to an AI model that can analyze both text and images."
        else:
            response = f"You said: '{prompt}'. "
            response += "This is a demo chatbot. In a real implementation, you would connect this to an AI model like OpenAI's GPT-4 Vision or similar."
        
        # Display the response
        st.markdown(response)
        
        # Add some example follow-up questions if it's the first message
        if len(st.session_state.messages) == 1:
            st.markdown("---")
            st.markdown("**Try asking things like:**")
            st.markdown("- Can you describe what's in this image?")
            st.markdown("- What colors are predominant in this picture?")
            st.markdown("- How many objects can you identify?")
    
    # Store the assistant's response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "type": "text"
    })

# Display chat instructions if no messages yet
if len(st.session_state.messages) == 0:
    st.info("👈 Upload an image from the sidebar or type a message below to start chatting!")

# Add a clear chat button in the sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()