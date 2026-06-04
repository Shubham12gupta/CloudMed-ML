# MedPredict AI 🏥🤖

An end-to-end Machine Learning web application designed to predict potential health conditions (such as Thyroid Disorder, Anemia, Hypertension, and more) based on real-time patient vitals. The system is fully deployed on AWS using a robust, scalable cloud architecture.

## 🚀 Live Demo
🔗 <img width="1726" height="969" alt="Screenshot 2026-05-30 223933" src="https://github.com/user-attachments/assets/0d473a6a-1319-43cd-a21f-d9cec448567e" />

<img width="1697" height="979" alt="Screenshot 2026-05-30 223948" src="https://github.com/user-attachments/assets/1577be1a-3e37-4ffe-8c67-0d15496d24d4" />

---

## 📸 Interface Preview
http://43.204.191.63

---

## ✨ Features
- **Multi-Condition Diagnostic Support:** Generates real-time confidence scores for multiple disorders (Thyroid, Anemia, Hypertension, Obesity, Infertility, Osteoporosis, Anxiety Disorder).
- **Dynamic Vitals Input:** Process critical patient metrics including Age, Gender, Blood Group, Temperature, and Pulse.
- **Production-Ready Architecture:** Designed with modern cloud infrastructure separating compute, storage, and database layers.

---

## 🛠️ Tech Stack & Architecture

### **Frontend & Backend**
- **Python:** Core programming language for data processing and modeling.
- **HTML5 / CSS3 / JavaScript:** For the interactive web interface.
- **Dockerfile:** Containerization for consistent environment deployment.

### **Cloud Infrastructure (AWS)**
- **Amazon EC2:** Hosts the web server and runs the real-time inference engine.
- **Amazon S3 (Simple Storage Service):** Acts as the model registry, securely storing serialized pipeline and machine learning model files (`.pkl`).
- **Amazon RDS (Relational Database Service):** Persists user input logs and historical ML prediction data for auditing and future model retraining.

---

## 📁 Repository Structure

```text
├── .github/                       # GitHub Actions / Workflows
├── appointment_module/ml_backend/ # Core ML prediction logic and scripts
├── localhost/                     # Frontend UI assets and HTML structure
├── .env                           # Environment variables (Database credentials, AWS Keys)
├── .gitignore                     # Securely ignores sensitive files (.env, cached files)
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
