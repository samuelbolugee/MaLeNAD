import streamlit as st
import joblib
import numpy as np
import spacy
import pandas as pd
from itertools import zip_longest
# Add dependency parsing visualization
from spacy import displacy
# Add language translation feature
from googletrans import Translator, LANGUAGES
# For WordCloud and others:
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load the models and vectorizer
spacy_nlp = spacy.load('en_core_web_sm') # to load an English-based model
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
# Get language codes and names
language_codes = list(LANGUAGES.keys())
language_names = list(LANGUAGES.values())

translator = Translator()

# Function to make predictions
@st.cache
def predict_fake_news(title, text):
    try:
        tf_idf = vectorizer.transform([f'{title} {text}'])
        prediction = model.predict(tf_idf)
        # Now, since the model is a PassiveAggressiveClassifier, we have to do things a little bit differently.
        decision_vals = model.decision_function(tf_idf)
        a = 1 - (1. / (1. + np.exp(decision_vals))) 
        b = -(a - 1)
        probs = np.stack((b, a), axis=1)
        probability = probs[0]
        return prediction, probability[0]
    except Exception as e:
        return "An error occurred during prediction:", str(e)

# Streamlit app
st.title("MaLeNAD: ML-Powered News Article Analysis and Fake News Detection App")
st.subheader('Analyse news articles and discern between fact and fiction, with this tool powered by machine learning!')

choices = ["Fake News Prediction", "Natural Language Processing", "Language Translation"]
st.sidebar.title("Welcome! :blush:")
choice = st.sidebar.selectbox("What do you want to do today?", choices)

if choice == "Fake News Prediction":
    st.info("You are now utilizing the Fake News Prediction module. Please note that predictions may not be 100% accurate.")
    # Input form for users
    title_input = st.text_input("Enter the title of the news article:")
    text_input = st.text_area("Enter the text of the news article:")

    # Make prediction when the user clicks the button
    if st.button("Predict"):
        with st.spinner("Processing, please wait..."):
            if title_input and text_input:
                prediction, probability = predict_fake_news(title_input, text_input)
                st.success("We can predict that this news article is {}, with {:.2f}% certainty.".format(prediction[0], probability[0] * 100))
            else:
                st.error("Please enter all required text. Thank you.")

elif choice ==  "Language Translation":
    st.info("You are now utilizing the Language Translation module, powered by Google Translate. Please note that translations may not be 100% accurate.")
    news_text = st.text_area("Enter your news article text:")
    target_language = st.selectbox("Select target language:", language_codes, format_func=lambda x: LANGUAGES[x])
    # Format the entire list of language names
    formatted_language_names = [LANGUAGES[code] for code in language_codes]
    if st.button("Translate!"):
        with st.spinner("Translating, please wait..."):
            if news_text:
                translated_text = translator.translate(text = news_text, dest=target_language).text
                st.subheader(f"Translated Text ({target_language}):")
                st.text_area("Translated Text:", translated_text)
            else:
                st.error(f"Please enter the required text. Thank you.")

else:
    st.info("You are now utilizing the Natural Language Processing module.")
    news_text = st.text_area("Enter your news article text:")
    tasks = ["Tokenization", "Dependency Parsing Visualization", "Named Entity Recognition", "Lemmatization", "Parts of Speech (POS) Tagging"]
    task_choice = st.selectbox("What do you want to do here?", tasks)
    if st.button("Process Text!"):
        with st.spinner("Processing, please wait..."):
            if news_text:
                doc = spacy_nlp(news_text)
                # Inside the "Natural Language Processing module" block
                if task_choice == "Dependency Parsing Visualization":
                    st.subheader("Dependency Parsing Visualization")
                    st.write("Here is the dependency parsing tree for the input text:")
                    rendition = displacy.render(doc, style="dep", options={'compact': True,'distance': 100})
                    st.image(rendition, width=100)
                elif task_choice == "Tokenization":
                    result = [token.text for token in doc]
                    st.json(result)
                elif task_choice == "Named Entity Recognition":
                    result = [(ent.text, ent.label_) for ent in doc.ents]
                    st.json(result)
                elif task_choice == "Lemmatization":
                    result = ["Token:{}, Lemma:{}".format(token.text, token.lemma_) for token in doc]
                    st.json(result)
                elif task_choice == "Parts of Speech (POS) Tagging":
                    result = ["Token:{}, POS:{}, Dependency:{}".format(token.text, token.tag_, token.dep_) for token in doc]
                    st.json(result)
            else:
                st.error("Please enter the required text. Thank you.")
   
    if st.button("Tabulate Information!"):
        with st.spinner("Tabulating, please wait..."):
            # Add a slider for selecting the number of rows to display
            num_rows = st.slider("Select the number of rows to display:", min_value=1, max_value=100, value=10)
            doc = spacy_nlp(news_text)
            tokens = [token.text for token in doc]
            entities = [(ent.label_) for ent in doc.ents]
            tags = [(token.tag_) for token in doc]

            # Create a list of entity labels for each token
            # This part of the code creates a list (entity_labels_for_tokens) where each element corresponds to a token. 
            # It iterates through the named entities (doc.ents) and assigns the entity label to the appropriate range of tokens.
            entity_labels_for_tokens = [""] * len(tokens)
            for ent in doc.ents:
                for i in range(ent.start, ent.end):
                    entity_labels_for_tokens[i] = ent.label_
            # Use zip_longest to ensure all columns have the same length
            zipped_data = zip_longest(tokens, entity_labels_for_tokens, tags, fillvalue="")
            # Create a DataFrame linking all three sets of information together:
            df = pd.DataFrame(zipped_data, columns = ["Tokens", "Lemma", "POS Tags"])
            st.dataframe(df.head(num_rows))


    if st.checkbox("Visualize a WordCloud"):
        with st.spinner("Visualizing, please wait..."):
            wc = WordCloud().generate(news_text)
            plt.imshow(wc, interpolation = "bilinear")
            plt.axis("off")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: none;
}

div[data-testid="stHorizontalBlock"] div[role="slider"] {
        background-color: #3498db;
    }

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ML and  ❤ by <a style='display: block; text-align: center;' href="https://www.github.com/samuelbolugee" target="_blank">S. B. Olugunna</a></p>
</div>

"""
st.markdown(footer,unsafe_allow_html=True)
