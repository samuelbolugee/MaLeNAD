# MaLeNAD
 An ML-powered news article analysis and fake news detection app, developed in Python, using Streamlit. Analyse news articles and discern between fact and fiction, with this tool powered by machine learning!
## Overview
Welcome to MaLeNAD, a machine learning-powered app for news article analysis and fake news detection. This tool provides functionalities for fake news prediction, natural language processing, and language translation.

## Features
- **Fake News Prediction:**
  - Input the title and text of a news article.
  - Click the "Predict" button to receive predictions on the authenticity of the news.

- **Language Translation:**
  - Input the news article text.
  - Select the target language from the provided options.
  - Click the "Translate!" button to get a translation of the article.

- **Natural Language Processing:**
  - Input the news article text.
  - Choose from various NLP tasks:
    - Tokenization
    - Dependency Parsing Visualization
    - Named Entity Recognition
    - Lemmatization
    - Parts of Speech (POS) Tagging

  - Click the "Process Text!" button to perform the selected NLP task.

  - Click the "Tabulate Information!" button to display tokenized information in a DataFrame.

  - Check the "Visualize a WordCloud" checkbox to generate and display a WordCloud.

## Installation
To run this app, you'll need to install the required dependencies. Create a virtual environment and install the dependencies using the following command:

```
pip install -r requirements.txt
```

## Usage
Run the app using the following command:

```
streamlit run fnapp.py
```
Access the app in your web browser at http://localhost:8501.
