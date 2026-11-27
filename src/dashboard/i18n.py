"""Korean translations for the VisionX dashboard."""

import streamlit as st

KO = {
    "Findings summary": "소견 요약",
    "Upload a chest X-ray, analyze it, and review the model's findings.": (
        "흉부 X선 영상을 업로드하고 분석하여 모델 결과를 확인하세요."
    ),
    "Enlarge image viewer": "영상 크게 보기",
    "Demo options": "데모 설정",
    "Show illustrative results": "예시 결과 표시",
    "Preview the interface without sending an image for analysis.": (
        "분석 요청 없이 예시 화면을 확인합니다."
    ),
    "Preview mode · Scores are illustrative and are not predictions for your image.": (
        "미리보기 모드 · 점수는 예시이며 업로드한 영상의 예측 결과가 아닙니다."
    ),
    "Upload image": "영상 업로드",
    "Please choose an image smaller than 10 MB.": "10 MB 미만의 영상을 선택하세요.",
    "Unable to read this image. Use a valid PNG/JPG smaller than 10 MB.": (
        "영상을 읽을 수 없습니다. 10 MB 미만의 올바른 PNG/JPG 파일을 사용하세요."
    ),
    "Analyze image": "영상 분석",
    "Analyzing image…": "영상 분석 중…",
    "Analysis took too long. Please try again.": "분석 시간이 초과되었습니다. 다시 시도하세요.",
    "The analysis service is unavailable. Please try again in a moment.": (
        "분석 서비스에 연결할 수 없습니다. 잠시 후 다시 시도하세요."
    ),
    "The service returned an incomplete result. Please try again.": (
        "불완전한 분석 결과가 반환되었습니다. 다시 시도하세요."
    ),
    "Loading example image…": "예시 영상 불러오는 중…",
    (
        "The example image could not load. Set PREVIEW_XRAY_PATH to "
        "a local X-ray file for offline preview."
    ): (
        "예시 영상을 불러올 수 없습니다. 오프라인 미리보기를 사용하려면 "
        "PREVIEW_XRAY_PATH에 로컬 X선 파일 경로를 설정하세요."
    ),
    (
        "Educational prototype. AI outputs are for reference and "
        "must not be used for clinical diagnosis."
    ): "교육용 프로토타입입니다. AI 결과는 참고용이며 임상 진단에 사용해서는 안 됩니다.",
    "Image viewer": "영상 보기",
    "Display": "보기 방식",
    "Original image": "원본 영상",
    "Illustrative overlay · Not model-generated": "예시 오버레이 · 모델이 생성한 결과가 아닙니다",
    "An explanation image is not available for this result.": (
        "이 결과에는 설명 영상이 제공되지 않습니다."
    ),
    "This colored overlay illustrates the layout. It is not Grad-CAM.": (
        "이 색상 오버레이는 화면 구성 예시이며 Grad-CAM 결과가 아닙니다."
    ),
    "Example findings": "예시 소견",
    "Findings that met their detection thresholds.": "점수가 소견별 탐지 임계값 이상인 항목입니다.",
    (
        "Detection thresholds are not configured. "
        "Model scores and the Grad-CAM overlay are shown below."
    ): "탐지 임계값이 설정되지 않았습니다. 모델 점수와 Grad-CAM 오버레이를 확인하세요.",
    "No finding met its detection threshold.": "탐지 임계값에 도달한 소견이 없습니다.",
    "This does not establish that the image is clinically normal.": (
        "이것이 영상의 임상적 정상을 의미하지는 않습니다."
    ),
    "Model details": "모델 정보",
    "DenseNet-121 + EfficientNet-B0 ensemble · 14 independent disease outputs": (
        "DenseNet-121 + EfficientNet-B0 앙상블 · 14개 질환별 독립 출력"
    ),
    "Grad-CAM source: DenseNet-121, fold 0": "Grad-CAM 생성 모델: DenseNet-121, 폴드 0",
    "Illustrative scores; no model was called.": "예시 점수이며 모델을 실행하지 않았습니다.",
    "Model version:": "모델 버전:",
    "Not supplied by API": "API에서 제공되지 않음",
    "Processing time (ms):": "처리 시간 (ms):",
    "All disease scores": "전체 질환별 점수",
    "Green · Below 0.30": "초록 · 0.30 미만",
    "Yellow · 0.30 to below 0.50": "노랑 · 0.30 이상 0.50 미만",
    "Red · 0.50 and above": "빨강 · 0.50 이상",
    "A dot marks a finding that met its own detection threshold.": (
        "소견 옆의 점은 해당 소견의 점수가 탐지 임계값 이상임을 나타냅니다."
    ),
    "Analysis results": "분석 결과",
    "Illustrative preview · Not model predictions": "예시 미리보기 · 모델 예측 결과가 아닙니다",
    "Analysis complete · Review findings and scores below.": (
        "분석 완료 · 아래에서 소견과 점수를 확인하세요."
    ),
    "Your results will appear here after you select Analyze image.": (
        "영상 분석 버튼을 누르면 여기에 결과가 표시됩니다."
    ),
    "Compare": "비교",
    "Original": "원본",
    "Explanation": "설명",
    "{name} · {width} × {height} pixels": "{name} · {width} × {height} 픽셀",
    (
        "[Sample X-ray: Mikael Häggström · CC0]({source}). "
        "Overlay and scores are illustrative, not model results for this image."
    ): (
        "[예시 X선 영상: Mikael Häggström · CC0]({source}). "
        "오버레이와 점수는 예시이며 이 영상에 대한 모델 결과가 아닙니다."
    ),
    (
        "Highlighted areas contributed to the DenseNet-121 score for the "
        "displayed disease. The overlay cannot confirm a diagnosis or "
        "precisely outline a lesion."
    ): (
        "강조된 영역은 표시된 질환의 DenseNet-121 점수에 영향을 준 부분입니다. "
        "이 오버레이만으로 질환을 진단하거나 병변의 정확한 경계를 확인할 수는 없습니다."
    ),
    "Disease to show in Grad-CAM": "Grad-CAM으로 확인할 질환",
    "Generating Grad-CAM…": "Grad-CAM 생성 중…",
    "Grad-CAM is unavailable for {disease}.": "{disease}의 Grad-CAM 히트맵을 표시할 수 없습니다.",
    "Weaker contribution": "기여도 낮음",
    "Stronger contribution": "기여도 높음",
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
