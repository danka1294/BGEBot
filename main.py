import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import openai

# Flask-App erstellen
app = Flask(__name__)

# OpenAI API-Schlüssel setzen
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# PDF-Dokument laden und verarbeiten
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

# Website-Inhalte scrapen
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([para.get_text() for para in paragraphs])
        return Document(page_content=text, metadata={"source": url})
    return None

# Setze deine Datenquellen hier
pdf_path = os.path.join(os.path.dirname(__file__), "BGEQuelle.pdf")
website_url = "https://fuereinander.jetzt"

# Lade PDF und Website-Daten
documents = load_pdf(pdf_path)
website_doc = scrape_website(website_url)

# Falls Website-Daten erfolgreich geladen wurden, füge sie hinzu
if website_doc:
    documents.append(website_doc)

# FAISS-Vektordatenbank erstellen
embedding_function = OpenAIEmbeddings(openai_api_key=openai.api_key)
db = FAISS.from_documents(documents, embedding_function)
db.save_local("faiss_index")

# Lade die FAISS-Datenbank
db = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True)

# GPT-4 Modell mit Retrieval nutzen
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

# API-Endpoint für den ChatBot erstellen
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data["message"]
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

# Flask-Server starten
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Port von Render verwenden
    app.run(host="0.0.0.0", port=port)
