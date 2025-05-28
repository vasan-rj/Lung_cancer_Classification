import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision import transforms, models
from skimage.feature import graycomatrix, graycoprops, hog
import pywt
from PIL import Image
from sklearn.preprocessing import StandardScaler
import os
import subprocess
import google.generativeai as genai
import pypandoc
import time

GEMINI_API_KEY = 'API_KEY'
genai.configure(api_key=GEMINI_API_KEY)

class VisionTransformerNet(nn.Module):
    def __init__(self, input_size, num_classes, patch_size=100, dim=128, depth=2, heads=4, mlp_dim=256, dropout=0.1):
        super(VisionTransformerNet, self).__init__()
        num_patches = input_size // patch_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_embedding = nn.Linear(patch_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_patches, self.patch_size)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :self.num_patches + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.fc(x)
        return x

densenet_model = models.densenet121(weights="IMAGENET1K_V1")
densenet_model = torch.nn.Sequential(*(list(densenet_model.children())[:-1]))
densenet_model.eval()

def extract_deep_features(image, return_activations=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    if return_activations:
        image.requires_grad_(True)
        features = densenet_model(image)
        return features.detach().squeeze().numpy().flatten(), features, image
    with torch.no_grad():
        features = densenet_model(image)
    return features.squeeze().numpy().flatten()

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    return np.array([contrast, dissimilarity])

def extract_wavelet_features(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return np.concatenate([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])

def extract_hog_features(image):
    image = cv2.resize(image, (128, 128))
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

FIXED_FEATURE_SIZE = 3000

def preprocess_ct_scan(img_path, scaler):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("Image could not be loaded.")
    img_pil = Image.open(img_path).convert('RGB')
    deep_features = extract_deep_features(img_pil)
    glcm_features = extract_glcm_features(img_gray)
    wavelet_features = extract_wavelet_features(img_gray)
    hog_features = extract_hog_features(img_gray)
    all_features = np.concatenate([deep_features, glcm_features, wavelet_features, hog_features])
    if all_features.shape[0] < FIXED_FEATURE_SIZE:
        all_features = np.pad(all_features, (0, FIXED_FEATURE_SIZE - all_features.shape[0]), mode='constant')
    else:
        all_features = all_features[:FIXED_FEATURE_SIZE]
    all_features = scaler.transform([all_features])
    return torch.tensor(all_features, dtype=torch.float32), img_gray

def predict_disease(img_path, model, scaler, device):
    features, _ = preprocess_ct_scan(img_path, scaler)
    features = features.to(device)
    class_names = np.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/densenet121_labelsc.npy", allow_pickle=True)
    unique_classes = np.unique(class_names)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = unique_classes[predicted.item()]
        confidence_score = confidence.item() * 100
    category_mapping = {
        'adenocarcinoma': 'Lung Cancer',
        'large_cell_carcinoma': 'Lung Cancer',
        'squamous_cell_carcinoma': 'Lung Cancer',
        'normal': 'Normal Lung'
    }
    mapped_category = category_mapping.get(predicted_class, 'Other')
    return mapped_category, predicted_class, confidence_score

def generate_bounding_box_image(img_path, predicted_class, device):
    img_pil = Image.open(img_path).convert('RGB')
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    height, width = img_gray.shape

    if predicted_class == 'normal':
        bounding_box_path = "/raid/home/posahemanth/Phase2/review2/bounding_box_image.jpg"
        cv2.imwrite(bounding_box_path, img_rgb)
        return bounding_box_path

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    densenet_model.eval()
    densenet_model.to(device)

    activations = densenet_model(input_tensor)
    activations.requires_grad_(True)

    linear = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(1024, 4)
    ).to(device)

    outputs = linear(activations)
    predicted_idx = outputs.argmax(dim=1).item()

    gradients = torch.autograd.grad(outputs[:, predicted_idx], activations, retain_graph=True)[0]

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    heatmap = cv2.resize(heatmap, (width, height))

    threshold = 0.8
    binary_heatmap = (heatmap > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        scale_factor = 0.5
        w = int(w * scale_factor)
        h = int(h * scale_factor)
        x = x + (w // 2)
        y = y + (h // 2)

        x1 = max(x - w // 2, 0)
        y1 = max(y - h // 2, 0)
        x2 = min(x + w // 2, width)
        y2 = min(y + h // 2, height)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)

    bounding_box_path = "/raid/home/posahemanth/Phase2/review2/bounding_box_image.jpg"
    cv2.imwrite(bounding_box_path, img_rgb)
    return bounding_box_path


def create_gemini_prompt(mapped_category, predicted_class, confidence_score):
    prompt = f"""
    You are a highly specialized radiology AI trained to interpret lung CT scans for diagnostic purposes. Based on the analysis of a lung CT scan, the AI model has provided the following prediction:

    - Broad Diagnostic Category: {mapped_category}
    - Specific Diagnosis (if applicable): {predicted_class}
    - Confidence Level: {confidence_score:.2f}%

    Using this information, classify the CT scan into one of these categories:
    - **Healthy Lung Tissue**: No abnormalities detected, indicating normal lung function and structure.
    - **Lung Cancer**: Evidence of malignancy such as nodules, masses, or irregular tissue growth.
    - **Other Abnormality**: Any condition not fitting the above (specify if possible).
    - **Insufficient Data**: If the prediction lacks clarity or detail for a definitive classification, explain why.

    Generate a detailed radiology report in a professional, medically precise tone. Include the following sections:

    1. **Observations**: Describe the likely imaging findings based on the predicted diagnosis (e.g., mass characteristics, tissue density, or absence of anomalies). Highlight specific features that support the classification.
    2. **Interpretation**: Provide a thorough interpretation of the CT scan, including:
       - **Conditions Identified**: Describe any medical conditions visible, such as pneumonia, lung cancer, interstitial lung disease, tuberculosis, pulmonary embolism, or other abnormalities, based on the model's prediction and typical imaging features.
       - **Severity and Extent**: Indicate the severity (e.g., mild, moderate, severe) and extent of any condition, including affected areas of the lungs (e.g., lobes, segments).
       - **Other Findings**: Mention significant findings not directly related to specific diseases but relevant to lung health (e.g., scars, cysts, pleural effusion), inferred from typical CT characteristics.
       - **Recommendations for Further Action**: Suggest follow-up actions based on the findings, such as additional imaging (e.g., PET-CT), biopsy, or clinical evaluation.
    3. **Diagnosis**: State the predicted condition with confidence level. For Lung Cancer, estimate the probable stage (e.g., I-IV) based on typical imaging traits, noting that further tests are needed for confirmation.
    4. **Treatment Recommendations**: Suggest evidence-based medical interventions (e.g., surgery, chemotherapy, or monitoring) aligned with standard guidelines for the diagnosed condition.
    5. **Nutritional Guidance**: Provide dietary advice, including beneficial foods to support recovery or health, and foods to limit or avoid that may worsen the condition.
    6. **Physical Activity**: Recommend appropriate exercises tailored to the patient’s condition, considering respiratory health and physical capacity.
    7. **Pharmacological Options**: If applicable, list standard medications or drug classes recommended for the condition (e.g., for symptom relief or treatment), based on medical protocols. Skip this if no specific medications apply or if it requires specialist input.

    Ensure the report is clear, actionable, and suitable for a physician’s review. Use precise medical terminology, avoid speculative or unsupported claims, and format the response in Markdown with headings (#) and bold text (*) where appropriate.
    """
    return prompt

def generate_radiology_report(mapped_category, predicted_class, confidence_score):
    prompt = create_gemini_prompt(mapped_category, predicted_class, confidence_score)
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    report = response.text
    return report

def main(img_path):
    global scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    scaler.fit(np.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/densenet121_featuresc.npy"))

    input_size = FIXED_FEATURE_SIZE
    num_classes = 4
    model = VisionTransformerNet(input_size, num_classes, patch_size=100, dim=128, depth=2, heads=4, mlp_dim=256)
    model.load_state_dict(torch.load("/raid/home/posahemanth/Phase2/review2/Processed_Data/model_weightsh.pth", map_location=device, weights_only=True))
    model.to(device)

    mapped_category, predicted_class, confidence_score = predict_disease(img_path, model, scaler, device)
    print(f"Predicted Category: {mapped_category}, Specific Prediction: {predicted_class}, Confidence: {confidence_score:.2f}%")

    detailed_report = generate_radiology_report(mapped_category, predicted_class, confidence_score)
    print("\nGenerated Radiology Report:")
    print(detailed_report)

    original_img_path = "/raid/home/posahemanth/Phase2/review2/original_image.jpg"
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(original_img_path, cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR))

    bounding_box_path = generate_bounding_box_image(img_path, predicted_class, device)

    enhanced_report = (
        "# Radiology Report\n"
        "## Patient Details\n"
        f"**Predicted Category**: {mapped_category}\n\n"
        f"**Specific Prediction**: {predicted_class}\n\n"
        f"**Confidence**: {confidence_score:.2f}%\n\n"
        "## Detailed Findings\n"
        f"{detailed_report}\n\n"
        "### Original CT Scan\n"
        f"![Original CT Scan]({original_img_path})\n\n\n"
        "### Predicted Cancer Region\n"
        f"![Predicted Cancer Region]({bounding_box_path})\n"
    )

    header_file = "header.tex"
    with open(header_file, "w") as f:
        f.write("\\usepackage{graphicx}\n\\usepackage[export]{adjustbox}\n")
    time.sleep(1)

    x = time.strftime("%d_%H-%M", time.localtime())
    output_file = f"/raid/home/posahemanth/Phase2/Reports/{x}_report_{predicted_class[0:3]}.pdf"
    temp_tex_file = f"/raid/home/posahemanth/Phase2/Reports/{x}_report_{predicted_class[0:3]}.tex"

    latex_content = pypandoc.convert_text(enhanced_report, 'latex', format='md', 
                                          extra_args=['--standalone', f'--include-in-header={os.path.abspath(header_file)}'])
    
    with open(temp_tex_file, "w") as f:
        f.write(latex_content)

    try:
        pdflatex_path = "/raid/home/posahemanth/texlive/bin/x86_64-linux/pdflatex"
        subprocess.run([pdflatex_path, temp_tex_file], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX to PDF: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        print("pdflatex not found. Ensure TeX Live is installed in /raid/home/posahemanth/texlive.")
        raise

    for ext in ['.tex', '.aux', '.log']:
        temp_file = f"/raid/home/posahemanth/Phase2/{x}_report_{predicted_class[0:3]}{ext}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists(header_file):
        os.remove(header_file)

    print(f"\nReport saved to '{output_file}'.")

if __name__ == "__main__":
    new_ct_scan_path = "/raid/home/posahemanth/Phase2/Data/test/squamous_cell_carcinoma/000131 (6).png"
    main(new_ct_scan_path)