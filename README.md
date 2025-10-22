# 📧 Spam Email Classifier

This project is a **machine learning model** that classifies emails as **Spam** or **Not Spam** using Python.  
It applies **Natural Language Processing (NLP)** and supervised learning techniques to analyze email text and predict spam probability.

---

## 🧠 Features
- Preprocessing of email text (cleaning, tokenization, vectorization)
- Trains a machine learning model (Naive Bayes / Logistic Regression)
- Takes user input to test predictions
- Easy to extend and improve for real-world datasets

---

## 🗂️ Dataset
- The dataset used is `spam.csv`
- It contains two columns:  
  - **v1** → label (`spam` / `ham`)  
  - **v2** → message text  

---

## ⚙️ Installation

### Prerequisites
Make sure you have **Python 3.8+** installed.

# Step 1: Clone the repository
git clone https://github.com/mohdaquib000001-cell/spam-email-classifier.git

# Step 2: Navigate to the project folder
cd spam-email-classifier

# Step 3: Install required dependencies
pip install pandas scikit-learn

# Step 4: Run the script
python spam_email.py
🧪 Sample Interaction
Enter your email: Congratulations! You won a free iPhone.
prediction: spam
