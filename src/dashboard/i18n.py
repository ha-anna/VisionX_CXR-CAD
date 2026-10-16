"""Display translations only; model/API labels remain unchanged."""

import streamlit as st

KO = {
    "Display and demo options": "화면 및 데모 설정",
    "Findings summary": "소견 요약",
    "Scores range from 0 to 1; they are not disease probabilities. A dot marks a detected finding. Detection uses disease-specific thresholds from the API.": "점수 범위는 0~1이며 질환의 확률이 아닙니다. 점 표시는 탐지된 소견을 나타냅니다. 탐지에는 API의 질환별 임계값을 사용합니다.",
    "VisionX\'s CXR-CAD System": "VisionX의 CXR-CAD 시스템",
    "Upload a chest X-ray, analyze it, and review the model's findings.": "흉부 X선 영상을 업로드하고 분석하여 모델 결과를 확인하세요.",
    "Enlarge image viewer": "영상 크게 보기",
    "Give images more space and place results below them.": "영상을 크게 표시하고 분석 결과를 아래에 배치합니다.",
    "Demo options": "데모 설정",
    "Show illustrative results": "예시 결과 표시",
    "Preview the interface without sending an image for analysis.": "분석 요청 없이 예시 화면을 확인합니다.",
    "Preview mode · Scores are illustrative and are not predictions for your image.": "미리보기 모드 · 점수는 예시이며 업로드한 영상의 예측 결과가 아닙니다.",
    "Upload image": "영상 업로드",
    "PNG or JPG · Up to 10 MB": "PNG 또는 JPG · 최대 10 MB",
    "Upload a chest X-ray": "흉부 X선 영상 업로드",
    "Please choose an image smaller than 10 MB.": "10 MB 미만의 영상을 선택하세요.",
    "Unable to read this image. Use a valid PNG/JPG smaller than 10 MB.": "영상을 읽을 수 없습니다. 10 MB 미만의 올바른 PNG/JPG 파일을 사용하세요.",
    "Analyze image": "영상 분석",
    "Analyzing image…": "영상 분석 중…",
    "Analysis took too long. Please try again.": "분석 시간이 초과되었습니다. 다시 시도하세요.",
    "The analysis service is unavailable. Please try again in a moment.": "분석 서비스에 연결할 수 없습니다. 잠시 후 다시 시도하세요.",
    "The service returned an incomplete result. Please try again.": "불완전한 분석 결과가 반환되었습니다. 다시 시도하세요.",
    "Loading example image…": "예시 영상 불러오는 중…",
    "The example image could not load. Set PREVIEW_XRAY_PATH to a local X-ray file for offline preview.": "예시 영상을 불러올 수 없습니다. 오프라인 미리보기를 사용하려면 PREVIEW_XRAY_PATH에 로컬 X선 파일 경로를 설정하세요.",
    "Educational prototype. AI outputs are for reference and must not be used for clinical diagnosis.": "교육용 프로토타입입니다. AI 결과는 참고용이며 임상 진단에 사용해서는 안 됩니다.",
    "The explanation image could not be displayed. Prediction scores are still available.": "설명 영상을 표시할 수 없습니다. 예측 점수는 확인할 수 있습니다.",
    "Image viewer": "영상 보기",
    "Display": "보기 방식",
    "Original image": "원본 영상",
    "Illustrative overlay · Not model-generated": "예시 오버레이 · 모델이 생성한 결과가 아닙니다",
    "Illustrative overlay": "예시 오버레이",
    "Grad-CAM explanation": "Grad-CAM 설명 영상",
    "An explanation image is not available for this result.": "이 결과에는 설명 영상이 제공되지 않습니다.",
    "How to read the explanation": "설명 영상 읽는 방법",
    "This colored overlay illustrates the layout. It is not Grad-CAM.": "이 색상 오버레이는 화면 구성 예시이며 Grad-CAM 결과가 아닙니다.",
    "Grad-CAM highlights regions influencing the selected disease score. It does not verify disease location.": "Grad-CAM은 선택된 질환의 점수에 영향을 준 영역을 강조합니다. 질환의 위치를 확정하는 것은 아닙니다.",
    "Example findings": "예시 소견",
    "Detected findings": "탐지된 소견",
    "Scores above their configured disease thresholds.": "설정된 질환별 임계값을 초과한 항목입니다.",
    "Detection thresholds are not configured. Model scores and the Grad-CAM overlay are shown below.": "탐지 임계값이 설정되지 않았습니다. 모델 점수와 Grad-CAM 오버레이를 확인하세요.",
    "This does not establish that the image is clinically normal.": "이것이 영상의 임상적 정상을 의미하지는 않습니다.",
    "Model details": "모델 정보",
    "DenseNet-121 · 14 independent disease outputs": "DenseNet-121 · 14개 질환별 독립 출력",
    "Illustrative scores; no model was called.": "예시 점수이며 모델을 실행하지 않았습니다.",
    "Model version:": "모델 버전:",
    "Not supplied by API": "API에서 제공되지 않음",
    "Processing time (ms):": "처리 시간 (ms):",
    "All disease scores": "전체 질환별 점수",
    "A dot marks a detected finding. Colors are score bands, not clinical severity. Detection uses disease-specific thresholds from the API.": "점 표시는 탐지된 소견을 나타냅니다. 색상은 점수 구간이며 임상적 중증도가 아닙니다. 탐지에는 API의 질환별 임계값을 사용합니다.",
    "Analysis results": "분석 결과",
    "Illustrative preview · Not model predictions": "예시 미리보기 · 모델 예측 결과가 아닙니다",
    "Analysis complete · Review findings and scores below.": "분석 완료 · 아래에서 소견과 점수를 확인하세요.",
    "Your results will appear here after you select Analyze image.": "영상 분석 버튼을 누르면 여기에 결과가 표시됩니다.",
    "Compare": "비교",
    "Original": "원본",
    "Explanation": "설명",
    "{name} · {width} × {height} pixels": "{name} · {width} × {height} 픽셀",
    "[Sample X-ray: Mikael Häggström · CC0]({source}). Overlay and scores are illustrative, not model results for this image.": "[예시 X선 영상: Mikael Häggström · CC0]({source}). 오버레이와 점수는 예시이며 이 영상에 대한 모델 결과가 아닙니다.",
    "VisionX GitHub repository": "VisionX GitHub 저장소",
}

DISEASES_KO = {
    "Atelectasis": "무기폐",
    "Cardiomegaly": "심비대",
    "Effusion": "흉수",
    "Infiltration": "폐 침윤",
    "Mass": "종괴",
    "Nodule": "결절",
    "Pneumonia": "폐렴",
    "Pneumothorax": "기흉",
    "Consolidation": "폐 경화",
    "Edema": "폐부종",
    "Emphysema": "폐기종",
    "Fibrosis": "폐섬유화",
    "Pleural_Thickening": "흉막 비후",
    "Hernia": "탈장",
}


def t(text, **values):
    translated = KO.get(text, text) if st.session_state.get("ui_language", "en") == "ko" else text
    return translated.format(**values) if values else translated


def disease_name(name):
    if st.session_state.get("ui_language", "en") == "ko":
        return DISEASES_KO.get(name, name.replace("_", " "))
    return name.replace("_", " ")
