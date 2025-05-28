import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import pydicom
import io



if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    # Set the API key from environment variable if not already set

with st.sidebar:
    st.title("ℹ️ Configuration")
    
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    
    st.info(
        "This tool provides AI-powered analysis of medical imaging data using "
        "advanced computer vision and radiological expertise."
    )
    # st.warning(
    #     "⚠DISCLAIMER: This tool is for educational and informational purposes only. "
    #     "All analyses should be reviewed by qualified healthcare professionals. "
    #     "Do not make medical decisions based solely on this analysis."
    # )

medical_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

doctor_recommendation_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

# New agent for specific doctor listings
local_doctors_agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not medical_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough.
"""

# Doctor Recommendation Query Template
doctor_recommendation_query = """
Based on the following medical analysis and diagnosis: 

{diagnosis_summary}

You are a healthcare referral specialist. Please recommend the most appropriate medical specialists and facilities that the patient should consider visiting based on their condition. Structure your response as follows:

### 1. Recommended Medical Specialties
- List the primary and secondary medical specialties best suited for this condition
- Explain why each specialty is relevant to this specific case

### 2. Typical Medical Professionals
- Provide examples of medical professionals (by specialty title) who typically treat this condition
- Include any subspecialties that might be particularly relevant

### 3. Questions to Ask When Selecting a Doctor
- List 3-5 important questions patients should ask potential doctors about this specific condition
- Include questions about experience, treatment approaches, and expected outcomes

### 4. Finding Local Specialists
- Suggest specific search terms patients can use to find relevant specialists
- Recommend reputable medical directories and hospital networks
- Provide information on certifications or qualifications to look for

### 5. Additional Resources
- Use DuckDuckGo to search for relevant patient support groups or associations
- Find and recommend trustworthy online resources for this condition
- Suggest any telehealth options that might be available

Format your response using clear markdown headers and bullet points. Keep your advice practical and actionable.
"""

# New query for specific doctor listings
local_doctors_query = """
Based on the following medical diagnosis and location information:

Diagnosis: {diagnosis_summary}
Location: {location}

You are a medical directory specialist. Use the DuckDuckGo search tool to find and list ACTUAL doctors in the patient's location who specialize in treating the diagnosed condition. Structure your response as follows:

### 📋 Recommended Specialists in {location}

For each doctor (find at least 5-8 if possible):

#### 1. Dr. [Full Name], [Specialization]
- **Clinic/Hospital**: [Name of practice/facility]
- **Address**: [Full address]
- **Contact**: [Phone number and/or email if available]
- **Expertise**: [Brief description of expertise relevant to the condition]
- **Website**: [If available]

#### 2. Dr. [Full Name], [Specialization]
...continue with same format for each doctor

### 🏥 Medical Centers & Clinics
- List 2-3 medical centers or specialized clinics in the area that treat this condition
- Include address, contact information, and website if available

### 💬 Appointment Tips
- Provide brief practical advice about scheduling appointments with these specialists
- Include information about typical wait times, insurance considerations, etc.

IMPORTANT NOTES:
1. ONLY include REAL doctors and medical facilities that actually exist in the specified location
2. Use DuckDuckGo to search for specialists in this location who treat this specific condition
3. Focus on providing accurate contact details and specializations
4. If the location information is too vague or you cannot find specific doctors, explicitly state that limitation and provide advice on how the patient can find local specialists
5. DO NOT invent or fabricate doctor information - only include verifiable specialists

Format your response using clear markdown headers and bullet points.
"""

st.title("🏥 Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for professional analysis")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()
doctor_recommendation_container = st.container()
local_doctors_container = st.container()  # New container for specific doctor listings

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Medical Image",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )
    
    # Add location input
    user_location = st.text_input("Your location (city, state/province, country)", 
                                 help="Optional: Providing your location helps find nearby medical specialists")

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
                
                # Resize image for display
                width, height = image.size
                aspect_ratio = width / height
                new_width = 500
                new_height = int(new_width / aspect_ratio)
                resized_image = image.resize((new_width, new_height))

                # Display image
                st.image(resized_image, caption="Uploaded Medical Image", use_container_width=True)

                # Analyze button (with unique key)
                analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True, key="analyze_button")

            except Exception as e:
                st.error(f"Error processing the image: {e}")
                analyze_button = False  # Avoids running analysis if error occurs

    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analyzing image... Please wait."):
                try:
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)

                    # Create AgnoImage object
                    agno_image = AgnoImage(filepath=temp_path)  # Ensure constructor matches

                    # Run analysis
                    response = medical_agent.run(query, images=[agno_image])

                    # Store the diagnosis summary for doctor recommendation
                    if "diagnosis_summary" not in st.session_state:
                        st.session_state.diagnosis_summary = response.content

                    # Display response
                    st.markdown("### 📋 Analysis Results")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Note: This analysis is generated by AI and should be reviewed by "
                        "a qualified healthcare professional."
                    )
                    
                    # Add find doctors button
                    st.session_state.show_doctor_recommendation = True
                    
                except Exception as e:
                    st.error(f"Analysis error: {e}")
    
    # Doctor recommendation section
    if "show_doctor_recommendation" in st.session_state and st.session_state.show_doctor_recommendation:
        with doctor_recommendation_container:
            st.markdown("### 👨‍⚕️ Find Medical Specialists")
            
            find_doctors = st.button("🔍 Find Recommended Specialists", type="primary", key="find_doctors_button")
            
            if find_doctors:
                with st.spinner("🔄 Finding medical specialists... Please wait."):
                    try:
                        # Format query with user location if provided
                        location_context = f"The patient is located in {user_location}. " if user_location else ""
                        final_query = doctor_recommendation_query.format(
                            diagnosis_summary=st.session_state.diagnosis_summary
                        )
                        
                        if location_context:
                            final_query = location_context + final_query + "\n\nIf possible, focus your recommendations on healthcare providers in or near the patient's location."
                        
                        # Run doctor recommendation
                        doctor_response = doctor_recommendation_agent.run(final_query)
                        
                        # Store response for local doctors lookup
                        st.session_state.doctor_recommendation = doctor_response.content
                        
                        # Display response
                        st.markdown("### 🏥 Recommended Medical Specialists")
                        st.markdown("---")
                        st.markdown(doctor_response.content)
                        st.markdown("---")
                        st.caption(
                            "Note: These recommendations are generated by AI and should be used as a starting point. "
                            "Always consult with your primary care physician for proper referrals."
                        )
                        
                        # Show local doctor search option only if location is provided
                        if user_location:
                            st.session_state.show_local_doctors = True
                        else:
                            st.warning("To find specific doctors in your area, please provide your location above.")
                            
                    except Exception as e:
                        st.error(f"Recommendation error: {e}")
    
    # Local doctors section - new section
    if "show_local_doctors" in st.session_state and st.session_state.show_local_doctors and user_location:
        with local_doctors_container:
            st.markdown("### 🔍 Find Specific Doctors Near You")
            
            find_local_doctors = st.button("📍 Find Doctors in Your Area", type="primary", key="find_local_doctors_button")
            
            if find_local_doctors:
                with st.spinner("🔄 Searching for specialists in your area... Please wait."):
                    try:
                        # Use diagnosis summary and location to find local doctors
                        final_local_query = local_doctors_query.format(
                            diagnosis_summary=st.session_state.diagnosis_summary,
                            location=user_location
                        )
                        
                        # Run local doctors search
                        local_doctors_response = local_doctors_agent.run(final_local_query)
                        
                        # Display response
                        st.markdown("### 📞 Contact These Specialists")
                        st.markdown("---")
                        st.markdown(local_doctors_response.content)
                        st.markdown("---")
                        st.caption(
                            "Note: Doctor listings are generated based on publicly available information. "
                            "Contact information may change over time. Always verify before making appointments."
                        )
                        
                        # Add expander with disclaimer
                        with st.expander("ℹ️ Important Information About Doctor Listings"):
                            st.markdown("""
                            - The doctor listings provided are based on publicly available information found through online searches.
                            - While we strive for accuracy, this information may not be up-to-date or complete.
                            - Always verify a doctor's credentials, specialty, and contact information before seeking treatment.
                            - This is not an endorsement of any specific healthcare provider.
                            - Your insurance coverage and network restrictions may limit which doctors you can see.
                            """)
                            
                    except Exception as e:
                        st.error(f"Local doctor search error: {e}")
else:
    st.info("👆 Please upload a medical image to begin analysis")