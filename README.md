
# Obecity Level Classification

## 📋 Project Overview

This project demonstrates end-to-end MLOps practices by building, training, and deploying a machine learning model to classify obesity levels. The system predicts one of seven obesity categories based on demographic data, eating habits, and lifestyle factors.

## Student Information
- **Name:** Steven Liementha
- **NIM:** 2702265370
- **Class:** LA09
- **Video Link:** 
- [https://binusianorg-my.sharepoint.com/personal/steven_liementha_binus_edu/_layouts/15/guestaccess.aspx?share=EmMxAwP_XW5Ov2YazTS6ptgBujE0XQCz3bZYDUIV2q-N4w&e=qxGGwh](https://binusianorg-my.sharepoint.com/personal/steven_liementha_binus_edu/_layouts/15/guestaccess.aspx?share=EmMxAwP_XW5Ov2YazTS6ptgBujE0XQCz3bZYDUIV2q-N4w&e=qxGGwh) -> OneDrive


## ⚙️ Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- **Python version:** `3.11.11` (recommended)

---

### Create and Activate Conda Environment

```bash
conda create --name mlops python=3.11

conda activate mlops

pip install -r requirements.txt

python model.py # for training model and export model

python backend.py #for run backend locally

streamlit run streamlit.py #to run streamlit locally
```

### Note : To Update requirements.txt
```sh
pip list --format=freeze > requirements.txt
```
