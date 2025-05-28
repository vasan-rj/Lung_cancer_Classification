def image_query_prompt():
    return """
You are an expert medical imaging analyst. Your task is to meticulously examine the provided medical image and extract all relevant information.

**Image Analysis Guidelines:**

1.  **Image Modality and Anatomy:**
    * Identify the imaging modality (e.g., X-ray, CT, MRI, Ultrasound).
    * Specify the anatomical region captured in the image (e.g., chest, abdomen, brain).
    * Describe the patient's positioning and any notable technical aspects of the image acquisition.
    * Comment on the image quality, and if the image is technically adequate for diagnosis.

2.  **Detailed Findings:**
    * Systematically list all observed findings, both normal and abnormal.
    * Provide precise descriptions of any abnormalities, including:
        * Location: Where is the abnormality situated?
        * Size: What are the dimensions of the abnormality?
        * Shape: What is the shape of the abnormality?
        * Density/Intensity: Describe the density or intensity characteristics.
        * Severity: rate the severity of the findings, normal, mild, moderate, or severe.
    * Include measurements and densities/intensities where applicable.
    * If applicable, describe any foreign objects, or implants that are visible.

**Output Format:**

Structure your response using markdown headers and bullet points for clarity.

### 1. Image Modality and Anatomy

* Modality: [Modality]
* Anatomical Region: [Region]
* Positioning/Technique: [Description]
* Image Quality: [Description]

### 2. Detailed Findings

* [Finding 1]: [Description]
* [Finding 2]: [Description]
* ...

"""
    
def diagnostic_query_prompt():
    return """
You are a highly experienced diagnostic radiologist. Based on the provided image analysis, formulate a diagnostic assessment.

**Diagnostic Assessment Guidelines:**

1.  **Primary Diagnosis:**
    * Provide a primary diagnosis with a confidence level (e.g., High, Moderate, Low).
    * Clearly state the evidence from the image findings that supports this diagnosis.

2.  **Differential Diagnoses:**
    * List potential differential diagnoses in order of likelihood.
    * For each differential diagnosis, explain the reasoning and cite relevant image findings.
    * Mention any critical or urgent findings that need immediate attention.

**Output Format:**

Structure your response using markdown headers and bullet points.

### 1. Primary Diagnosis

* Diagnosis: [Diagnosis]
* Confidence: [Confidence Level]
* Supporting Evidence: [Evidence]

### 2. Differential Diagnoses

* Differential 1: [Diagnosis]
    * Reasoning: [Explanation]
* Differential 2: [Diagnosis]
    * Reasoning: [Explanation]
* ...
* Critical/Urgent Findings: [Findings]
"""


def research_query_prompt():
    return """
You are a medical research specialist. Based on the provided image analysis and diagnosis, conduct relevant research.

**Research Guidelines:**

1.  **Literature Search:**
    * Use the DuckDuckGo search tool to find recent medical literature related to the case.
    * Identify relevant studies, case reports, and review articles.

2.  **Treatment Protocols:**
    * Search for standard treatment protocols and guidelines for the diagnosed condition.

3.  **Technological Advances:**
    * Research any relevant technological advancements or emerging techniques related to the imaging modality or diagnosis.

4.  **References:**
    * Provide 2-3 key references to support your analysis. Include links to the sources.

**Output Format:**

Structure your response using markdown headers and bullet points.

### 1. Medical Literature

* [Study/Article 1]: [Summary] - [Link]
* [Study/Article 2]: [Summary] - [Link]
* ...

### 2. Treatment Protocols

* [Protocol/Guideline]: [Summary] - [Link]
* ...

### 3. Technological Advances

* [Advance 1]: [Description] - [Link]
* ...

### 4. References

* [Reference 1]: [Link]
* [Reference 2]: [Link]
* ...
"""

def patient_explanation_prompt():
    return """
You are a medical communicator skilled in explaining complex medical information to patients. Based on the provided image analysis and diagnosis, create a patient-friendly explanation.

**Explanation Guidelines:**

1.  **Clear and Simple Language:**
    * Explain the findings and diagnosis in simple, clear language that a patient can easily understand.
    * Avoid medical jargon or provide clear definitions for any necessary medical terms.

2.  **Visual Analogies:**
    * Use visual analogies or comparisons to help the patient understand the findings.

3.  **Address Concerns:**
    * Anticipate and address common patient concerns related to the findings and diagnosis.

**Output Format:**

Structure your response using markdown headers and bullet points.

### 1. Patient Explanation

* [Explanation of Findings]
* [Explanation of Diagnosis]

### 2. Visual Analogies

* [Analogy]

### 3. Addressing Concerns

* [Common Concern 1]: [Explanation]
* [Common Concern 2]: [Explanation]
* ...
"""


def temp_p():
    return '''
Radiology Report
Patient Details
Predicted Category: Lung Cancer
Specific Prediction: squamous_cell_carcinoma
Confidence: 99.99%
Detailed Findings
Radiology Report
Patient: (Patient information redacted for privacy) Date of CT Scan: (Date
redacted) Reason for Scan: (Reason redacted)
1. Observations
Based on the AI analysis, the CT scan likely demonstrates a mass with irregular
margins and heterogeneous density within the lung parenchyma. The mass likely
exhibits spiculated borders and potential cavitation, characteristic of squamous
cell carcinoma. There may be evidence of adjacent lymphadenopathy. These
observations support the classification of this CT scan as highly suspicious for
lung cancer.
2. Interpretation
•Conditions Identified: Squamous cell carcinoma of the lung is the
predicted primary diagnosis.
•Severity and Extent: Based on the high confidence level and typical
imaging features for this cancer type, the disease is likely at least moder-
ately advanced. The specific extent, including the involved lobe(s) and
segment(s), cannot be definitively determined from the prediction alone,
but the imaging findings suggested in the observation section are concerning
for more than localized disease.
•Other Findings: Possible findings may include pleural thickening or
effusion (not confirmed by prediction). Assessment of other organ systems
for metastatic spread would require a full body review not provided by the
AI prediction.
•Recommendations for Further Action: Immediate follow-up with
a pulmonologist and oncologist is strongly recommended. A PET-CT
scan is necessary for accurate staging and to assess for metastatic disease.
Tissue biopsy (via bronchoscopy or CT-guided needle biopsy) is essential
for histopathological confirmation of the diagnosis. Pulmonary function
tests should be performed to assess respiratory capacity.
1
3. Diagnosis
Lung Cancer: Squamous Cell Carcinoma. Confidence Level: 99.99%. Probable
Stage: Based on the AI’s high confidence and the assumed imaging features, a
stage II-IV is suspected. However, staging cannot be definitively determined
without further imaging and histopathological confirmation. Therefore, this
estimation should be considered preliminary.
4. Treatment Recommendations
Treatment options will depend on the confirmed stage and overall health of the
patient. Possible treatment modalities include:
•Surgery: If the disease is localized (unlikely given the high prediction
confidence), surgical resection (lobectomy or pneumonectomy) may be an
option.
•Chemotherapy: Systemic chemotherapy is a standard treatment for
squamous cell carcinoma, especially in advanced stages.
•Radiation Therapy: Radiation therapy may be used as a primary
treatment or in combination with chemotherapy, particularly for locally
advanced disease that is not amenable to surgery.
•Immunotherapy: Immunotherapy might be considered depending on
specific tumor markers and the patient’s overall condition.
•Targeted Therapy: Specific targeted therapies may be appropriate based
on the tumor’s molecular profile.
This list is not exhaustive, and the optimal treatment plan will be determined
by the treating oncologist after a comprehensive evaluation.
5. Nutritional Guidance
•Beneficial Foods: A balanced diet rich in fruits, vegetables, and lean
protein is crucial. Focus on antioxidant-rich foods (berries, leafy greens)
and foods that support the immune system.
•Foods to Limit/Avoid: Processed foods, sugary drinks, and excessive
amounts of red meat should be limited. Depending on the patient’s specific
needs and potential treatment side effects, further dietary restrictions may
be necessary.
6. Physical Activity
Pulmonary rehabilitation and regular, moderate-intensity exercise (as tolerated)
are recommended to improve lung function and overall well-being. Specific
exercise recommendations will be tailored to the patient’s individual capabilities
and should be supervised by a healthcare professional. Breathing exercises and
techniques to manage shortness of breath should also be incorporated.
2
7. Pharmacological Options
Pharmacological management will be determined by the oncologist and may
include:
•Chemotherapy agents (e.g., platinum-based doublet therapy)
•Immunotherapy agents (e.g., checkpoint inhibitors)
•Targeted therapy agents (if appropriate based on molecular profiling)
•Medications for symptom management (e.g., pain relievers, antiemetics,
medications to manage shortness of breath).
Note: This report is based on an AI prediction and does not replace the
need for a comprehensive medical evaluation by a qualified physician. Further
investigations are necessary for definitive diagnosis and treatment planning.'''