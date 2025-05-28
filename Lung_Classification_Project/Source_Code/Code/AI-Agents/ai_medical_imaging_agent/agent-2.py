import os
import tempfile
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import pydicom
import io

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

# ... (Sidebar code remains the same) ...

medical_agent = Agent(
    model=Gemini(
        id="gemini-pro-vision", # Using pro vision for better image analysis.
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

# ... (API key check) ...

# Medical Analysis Query (Slightly modified)
query = """
You are a highly skilled medical imaging expert... (same as before) ...
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# ... (Container setup) ...

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                if uploaded_file.name.lower().endswith(".dicom"):
                    ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
                    image = PILImage.fromarray(ds.pixel_array)
                else:
                    image = PILImage.open(uploaded_file)
                width, height = image.size
                aspect_ratio = width / height
                new_width = 500
                new_height = int(new_width / aspect_ratio)
                resized_image = image.resize((new_width, new_height))
                st.image(resized_image, caption="Uploaded Medical Image", use_container_width=True)
                analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
            except Exception as e:
                st.error(f"Image loading error: {e}")
                analyze_button = False
    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
                        resized_image.save(temp_file.name)
                        agno_image = AgnoImage(filepath=temp_file.name)
                        response = medical_agent.run(query, images=[agno_image])
                        st.markdown("### 📋 Analysis Results")
                        st.markdown("---")
                        st.markdown(response.content)
                        st.markdown("---")
                        st.caption("Note: This analysis... (same as before) ...")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
else:
    st.info("👆 Please upload a medical image to begin analysis")