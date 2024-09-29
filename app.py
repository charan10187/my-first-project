from flask import Flask, render_template, request, jsonify
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data
with open('chatbot.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Tokenize the document
sent_tokens = nltk.sent_tokenize(raw_doc)  
word_tokens = nltk.word_tokenize(raw_doc)

# Initialize the WordNet Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Lemmatize tokens
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Normalize text by removing punctuation and lemmatizing
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greetings
GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREET_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)

# Generate response using TF-IDF and cosine similarity
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    
    sent_tokens.remove(user_response)
    return robo_response

# Route to load the frontend HTML page
@app.route("/")
def index():
    return render_template("index.html")

# API route to get a response from the chatbot
@app.route("/get_response", methods=["POST"])
def get_bot_response():
    user_input = request.json.get("user_input")
    user_input = user_input.lower()
    
    if user_input == 'bye':
        return jsonify({"bot_response": "Goodbye! Take care "})
    
    if user_input in ('thanks', 'thank you'):
        return jsonify({"bot_response": "You are welcome."})
    
    bot_response = greet(user_input)
    if bot_response is None:
        bot_response = response(user_input)
    
    return jsonify({"bot_response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
