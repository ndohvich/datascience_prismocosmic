from flask import Flask, render_template, request, jsonify
import nltk
import spacy
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM

# Download NLTK data
nltk.download('punkt')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize tokenizer with appropriate settings
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize Huggingface transformers pipeline
chatbot = pipeline("conversational", model=model, tokenizer=tokenizer)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['message']

    # Use NLTK for tokenization
    tokens = nltk.word_tokenize(user_input)

    # Use SpaCy for NER and POS tagging
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]

    # Create a Conversation object for the transformer model
    conversation = Conversation(user_input)

    # Use transformers to get the chatbot response
    response = chatbot(conversation)
    bot_response = response.generated_responses[0]

    return jsonify({
        'response': bot_response,
        'tokens': tokens,
        'entities': entities,
        'pos_tags': pos_tags
    })


if __name__ == "__main__":
    app.run(debug=True)

