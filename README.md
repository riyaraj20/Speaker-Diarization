# Speaker Detection and Timestamping Tool

## 📌 Project Overview
This project focuses on **Speaker Diarization for Indian Languages**, specifically Hindi and Bhojpuri.  
The system identifies **who spoke when** in a given audio file and generates speaker-wise timestamps.

Most existing diarization systems are trained on English datasets and perform poorly on Indian language audio. This project aims to adapt and evaluate speaker diarization models for regional Indian languages.

---

## 🎯 Objective
- Build an end-to-end system for speaker diarization on Hindi/Bhojpuri audio  
- Generate speaker-wise timestamps automatically  
- Fine-tune a pretrained model for Indian language speech  
- Evaluate performance using Diarization Error Rate (DER)

---

## 🚩 Problem Statement
Existing diarization systems face challenges when applied to Indian languages due to:

- Speaker overlap  
- Dialectal variations  
- Informal speech patterns  
- Lack of labeled Indian language datasets  

This project attempts to bridge that gap by creating a custom dataset and fine-tuning models for better performance.

---

## 🛠 Proposed Solution
An end-to-end pipeline that:

- Takes conversational audio as input  
- Detects multiple speakers  
- Generates timestamps for each speaker segment  
- Supports Hindi and Bhojpuri audio  
- Evaluates output using DER metric  

---

## 📂 Dataset Details
- **Languages:** Hindi & Bhojpuri  
- **Total Audio Files:** 89  
- **Type:** Multi-speaker conversational audio  
- **Annotation Format:** RTTM  
- **Split:** Train / Development / Test  

Manual annotations were performed to create a high-quality evaluation dataset.

---

## 🧠 Model & Approach
- Used a **pretrained speaker segmentation model**  
- Fine-tuned on our custom Indian language dataset  
- Training performed using **PyTorch Lightning**  
- Focused on segmentation adaptation rather than training from scratch  

---

## 📊 Evaluation Metric

**Diarization Error Rate (DER)** was used to measure performance.

### Results:
- Total evaluated speech: 230 seconds  
- Correct speech detection: ~89%  
- Errors mainly due to overlap and noise  
- Final DER: ~59%

---

## 🛠 Tools and Technologies Used

| Tool / Library | Purpose |
|----------------|---------|
| Python | Programming Language |
| PyTorch | Model Training |
| PyTorch Lightning | Training Framework |
| PyAnnote Audio | Speaker Diarization |
| SpeechBrain | Speaker Embeddings |
| Torchaudio | Audio Processing |
| RTTM Format | Manual Annotations |
| pyannote.metrics | DER Computation |
| TensorBoard | Training Monitoring |
| Matplotlib & Pandas | Analysis & Visualization |
| Streamlit | User Interface |

---

## ⚙ Features

- Multi-speaker detection  
- Automatic timestamp generation  
- Support for Hindi & Bhojpuri  
- Ground truth vs prediction comparison  
- Performance evaluation using DER  
- User-friendly interface using Streamlit  

---

## 📌 How to Run the Project

1. Clone the repository  
```bash
git clone <repository-link>

Install dependencies
pip install -r requirements.txt

Run the application

streamlit run app.py


Upload an audio file and view diarization results.
