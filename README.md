# 🐦 Twitter Sentiment Analysis

A complete Machine Learning project for analyzing sentiment in tweets using Natural Language Processing (NLP) and supervised learning algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Features](#features)

---

## 🎯 Overview

This project performs sentiment analysis on Twitter data to classify tweets into three categories:
- **Positive** (😊): Favorable opinions and support
- **Neutral** (😐): Objective or mixed statements  
- **Negative** (😞): Critical or opposing views

The project includes:
- Data exploration and visualization
- Training and evaluation of multiple ML models
- Interactive web application for real-time predictions
- Comprehensive analysis and reporting

---

## 📊 Dataset

**Source:** Twitter Data about Modi  
**Size:** 162,969 tweets  
**Columns:**
- `clean_text`: Pre-processed tweet text
- `category`: Sentiment label (-1: Negative, 0: Neutral, 1: Positive)

**Distribution:**
- Positive: 72,249 tweets (44.3%)
- Neutral: 55,211 tweets (33.9%)
- Negative: 35,509 tweets (21.8%)

**Statistics:**
- Average tweet length: 124 characters
- Shortest tweet: 1 character
- Longest tweet: 274 characters

---

## 🤖 Models

### 1. Logistic Regression (Primary Model) ✅

**Hyperparameters:**
- max_iter: 500
- class_weight: 'balanced'
- solver: 'lbfgs'

**Performance:**
- **Accuracy:** 90.72%
- **F1-Score:** 0.9071

**Detailed Metrics:**

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.8381    | 0.8529 | 0.8454   | 7,102   |
| Neutral  | 0.8940    | 0.9728 | 0.9318   | 11,042  |
| Positive | 0.9564    | 0.8837 | 0.9186   | 14,450  |

### 2. Random Forest

**Hyperparameters:**
- n_estimators: 200
- max_depth: 20
- class_weight: 'balanced'

**Performance:**
- **Accuracy:** 73.73%
- **F1-Score:** 0.7391

---

## 📈 Results

### Model Comparison

| Model                | Accuracy | F1-Score | Training Time |
|---------------------|----------|----------|---------------|
| Logistic Regression | 90.72%   | 0.9071   | ~10 seconds   |
| Random Forest       | 73.73%   | 0.7391   | ~7 minutes    |

**Winner:** Logistic Regression 🏆

### Key Findings

1. **Logistic Regression significantly outperformed Random Forest** (17% higher accuracy)
2. **Best at detecting Positive sentiments** (95.64% precision)
3. **Excellent Neutral detection** (97.28% recall)
4. **Fast training and prediction** times make it ideal for production

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualizations
- `streamlit` - Web application framework

---

## 🚀 Usage

### Option 1: Train Models from Scratch

Run the complete training pipeline:

```bash
python twitter_sentiment_analysis.py
```

This will:
1. Load and explore the data
2. Train Logistic Regression model
3. Train Random Forest model
4. Generate comparison visualizations
5. Save all outputs to the current directory

**Output Files:**
- `sentiment_distribution.png` - Distribution of sentiment classes
- `cm_logistic_regression.png` - Confusion matrix for LR
- `cm_random_forest.png` - Confusion matrix for RF
- `models_comparison.png` - Performance comparison chart

### Option 2: Run Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**App Features:**
- 🏠 **Home:** Overview and statistics
- 🎯 **Predict:** Real-time sentiment prediction
  - Single tweet analysis
  - Batch processing
  - Confidence scores
- 📊 **Train Model:** Upload data and train custom models
- ℹ️ **About:** Project information and metrics

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (Free & Easy)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/twitter-sentiment.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Your app will be live at:**
```
https://yourusername-twitter-sentiment.streamlit.app
```

### Deploy to Heroku

1. **Create `Procfile`:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. **Create `setup.sh`:**
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. **Deploy:**
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

### Deploy with Docker

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

2. **Build and Run:**
```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

---

## ✨ Features

### Data Analysis
- ✅ Comprehensive data exploration
- ✅ Statistical analysis and visualizations
- ✅ Class distribution analysis
- ✅ Text length statistics

### Machine Learning
- ✅ TF-IDF vectorization (5,000 features)
- ✅ Multiple model training and comparison
- ✅ Hyperparameter optimization
- ✅ Model evaluation with multiple metrics

### Web Application
- ✅ Modern, responsive UI
- ✅ Real-time predictions
- ✅ Batch processing
- ✅ Confidence scores visualization
- ✅ Model training interface
- ✅ CSV export functionality
- ✅ Interactive charts

### Visualization
- ✅ Confusion matrices
- ✅ Performance comparison charts
- ✅ Probability distributions
- ✅ Sentiment distribution plots

---

## 🎓 How It Works

### 1. Text Preprocessing
- Tweets are already cleaned in the dataset
- Text is lowercase
- URLs, mentions, special characters removed

### 2. Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Min document frequency: 2

### 3. Model Training
- **Algorithm:** Logistic Regression
- **Regularization:** Balanced class weights
- **Optimization:** LBFGS solver
- **Max iterations:** 500

### 4. Prediction
```
Input Tweet → TF-IDF Vectorization → Model → Probabilities → Sentiment Label
```

---

## 📊 Performance Metrics Explained

### Accuracy
Overall percentage of correct predictions: **90.72%**

### Precision
Of all tweets predicted as a class, how many were actually that class?
- Negative: 83.81%
- Neutral: 89.40%
- Positive: 95.64%

### Recall
Of all actual tweets in a class, how many were correctly predicted?
- Negative: 85.29%
- Neutral: 97.28%
- Positive: 88.37%

### F1-Score
Harmonic mean of precision and recall:
- Negative: 0.8454
- Neutral: 0.9318
- Positive: 0.9186

---

## 🔧 Customization

### Change Model Parameters

Edit `twitter_sentiment_analysis.py`:

```python
# Logistic Regression
lr_model = LogisticRegression(
    max_iter=500,      # Increase for more iterations
    C=1.0,             # Regularization strength
    class_weight='balanced'
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,    # Number of features
    ngram_range=(1, 2),   # Unigrams and bigrams
    min_df=2              # Minimum document frequency
)
```

### Add New Models

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# SVM
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Dataset Not Found

**Solution:**
Place `Twitter_Data.csv` in the project root directory or update the path in the script:
```python
df = pd.read_csv('Twitter_Data.csv')
```

### Issue: Streamlit App Won't Start

**Solution:**
```bash
# Check if Streamlit is installed
streamlit --version

# Reinstall if needed
pip install streamlit==1.31.0
```

---

## 🚀 Future Enhancements

- [ ] Add BERT/Transformer models
- [ ] Implement real-time Twitter streaming
- [ ] Multi-language support
- [ ] Sentiment intensity scoring
- [ ] Topic modeling integration
- [ ] User authentication
- [ ] API endpoint creation
- [ ] Docker containerization

---

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset source: Twitter API
- Tools: Scikit-learn, Streamlit, Pandas
- Inspiration: Natural Language Processing community

---
## 👤 Author

**Ahmed Salama**
*Data Scientist*

[LinkedIn](https://www.linkedin.com/in/ahmedsalamaa00/) | [GitHub](https://github.com/ahmedsalama00) | [Portfolio](https://ahmedsalama00.github.io/Ahmed)
