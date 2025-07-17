import os
import streamlit as st
from dotenv import load_dotenv
import pf
import json

# Load environment variables
load_dotenv()

title = os.getenv("TITLE", "Contoso Entity Recognition")
logo = os.getenv("LOGO_URL", "images/msft_logo.png")
endpoint = os.getenv("PROMPTFLOW_ENDPOINT")
api_key = os.getenv("PROMPTFLOW_KEY")
feedback_endpoint = os.getenv("FEEDBACK_ENDPOINT")

# Initialize session state for result
if "output" not in st.session_state:
    st.session_state.output = ""

# Layout for the app
col1, col2 = st.columns([4, 1])
with col1:
    st.title(title)
with col2:    
    st.image(logo, width=50)

# Input fields
entity_type = st.text_input("Entity Type")
input_text = st.text_input("Input Text")

if st.button("Submit"):
    # Prepare data for processing
    data = {
        "entity_type": entity_type,
        "text": input_text
    }
    result = pf.process_with_promptflow(data, endpoint, api_key)

    # Decode and parse the model result
    decoded_string = result.decode('utf-8')
    data = json.loads(decoded_string)
    st.session_state.output = ", ".join(data["entities"])

# Model output container
output_container = st.container()

# Display the model output from session state
with output_container:
    st.markdown(f"**Model output:** {st.session_state.output}")

# Sentiment feedback container
feedback_container = st.container()
with feedback_container:
    st.markdown("How did we do?")
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs")
    if selected is not None:
        # Display selected sentiment
        st.markdown(f"You selected: {sentiment_mapping[selected]}")
        
        # Send feedback
        feedback_data = {"feedback": "thumbsup" if selected == 1 else "thumbsdown"}
        pf.feedback(feedback_data, api_key, feedback_endpoint)
