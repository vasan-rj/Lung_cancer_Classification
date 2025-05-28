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
from prompts import temp_p
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

if "image_memory" not in st.session_state:
    st.session_state.image_memory = []
if "diagnostic_memory" not in st.session_state:
    st.session_state.diagnostic_memory = []
if "research_memory" not in st.session_state:
    st.session_state.research_memory = []
if "patient_memory" not in st.session_state:
    st.session_state.patient_memory = []
if "combined_response" not in st.session_state:
    st.session_state.combined_response = ""
if "agno_image" not in st.session_state:
    st.session_state.agno_image = None

with st.sidebar:
    st.title("ℹ️ Configuration")
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input("Enter your Google API Key:", type="password")
        st.caption("Get your API key from [Google AI Studio](https://aistudio.google.com/apikey) 🔑")
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    st.info("This tool provides AI-powered analysis of medical imaging data using advanced computer vision and radiological expertise.")

# Agent Initialization
def create_agent():
    return Agent(
        model=Gemini(id="gemini-1.5-flash", api_key=st.session_state.GOOGLE_API_KEY),
        tools=[DuckDuckGoTools()],
        markdown=True,
    ) if st.session_state.GOOGLE_API_KEY else None

image_agent = create_agent()
diagnostic_agent = create_agent()
research_agent = create_agent()
patient_agent = create_agent()

# Agent Queries
image_query = """You are a medical imaging expert. Analyze the image and provide:
the given images will be having any one of these disease 1. adenocarcinoma , 2. squamous cell carcinoma, 3. large cell carcinoma  4. normal lung 
after analyzing the image provide the following information
- Image Type & Region
- Key Findings
"""

diagnostic_query = """You are a diagnostic expert. Given the findings, provide:
- Diagnostic Assessment
"""

research_query = """You are a medical researcher. Given the findings, provide:
- Research Context
"""

patient_query = """You are a medical communicator. Given the findings, provide:
- Patient-Friendly Explanation
"""

st.title("Multi-Agent Medical Imaging Diagnosis")
st.write("Upload a medical image for professional analysis")

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()
question_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader("Upload Medical Image", type=["jpg", "jpeg", "png", "dicom"], help="Supported formats: JPG, JPEG, PNG, DICOM")

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                if uploaded_file.name.lower()=='squamous_cell_carcinoma.png':
                    temp_text=temp_p()
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
                analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True, key="analyze_button")
            except Exception as e:
                st.error(f"Error processing the image: {e}")
                analyze_button = False

    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
                        resized_image.save(temp_file.name)
                        st.session_state.agno_image = AgnoImage(filepath=temp_file.name)

                        # Run agents
                        image_response = image_agent.run(image_query, images=[st.session_state.agno_image]).content
                        st.session_state.image_memory.append(image_response)

                        diagnostic_response = diagnostic_agent.run(f"{diagnostic_query} {' '.join(st.session_state.image_memory)}").content
                        st.session_state.diagnostic_memory.append(diagnostic_response)

                        research_response = research_agent.run(f"{research_query} {' '.join(st.session_state.diagnostic_memory)}").content
                        st.session_state.research_memory.append(research_response)

                        patient_response = patient_agent.run(f"{patient_query} {' '.join(st.session_state.research_memory)}").content
                        st.session_state.patient_memory.append(patient_response)

                        # Combine results
                        st.session_state.combined_response = f"""
### 1. Image Type & Region and Key Findings:
{image_response}

### 2. Diagnostic Assessment:
{diagnostic_response}

### 3. Research Context:
{research_response}

### 4. Patient-Friendly Explanation:
{patient_response}
"""
                        st.markdown("### 📋 Combined Analysis Results")
                        st.markdown("---")
                        st.markdown(st.session_state.combined_response)
                        st.markdown("---")
                        st.caption("Note: This analysis is generated by AI and should be reviewed by a qualified healthcare professional.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
    with question_container:
        if st.session_state.combined_response:
            question = st.text_input("Ask a question about the report or image:")
            if question:
                if "image" in question.lower() or "find" in question.lower() or "type" in question.lower():
                    response = image_agent.run(f"{question} Report: {st.session_state.combined_response}", images=[st.session_state.agno_image]).content
                elif "diagnosis" in question.lower() or "diagnose" in question.lower():
                    response = diagnostic_agent.run(f"{question} Report: {st.session_state.combined_response}").content
                elif "research" in question.lower() or "treatment" in question.lower():
                    response = research_agent.run(f"{question} Report: {st.session_state.combined_response}").content
                elif "patient" in question.lower() or "explain" in question.lower() :
                    response = patient_agent.run(f"{question} Report: {st.session_state.combined_response}").content
                else :
                    response = image_agent.run(f"{question} Report: {st.session_state.combined_response}", images=[st.session_state.agno_image]).content

                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {response}")
else:
    st.info("👆 Please upload a medical image to begin analysis")