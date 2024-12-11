from flask import Flask, request, jsonify
import os
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
from langchain_community.document_loaders import PyPDFLoader
from werkzeug.utils import secure_filename
from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain_community.chat_models import ChatOpenAI # Import OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate
import openai
import difflib
import pdfplumber

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chromaDB")  
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./upload")

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer and don't share the related source document links.
    2. If you find the answer, write the answer in a concise way with five sentences maximum and also share the related document links.
{context}
 - -
Answer the question based on the above context: {question}
"""

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize OpenAI chat model
model = ChatOpenAI()

def split_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    return text_splitter.split_documents(docs_before_split)

def load_file_into_db(file_path):
    docs_after_split = split_documents(file_path)
    
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
        db.add_documents(docs_after_split)
        return db
    else:
        return Chroma.from_documents(docs_after_split, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Chunk and load the file into chroma db
        load_file_into_db(file_path)
        
        return jsonify({"message": "File uploaded and processed successfully"}), 200

@app.route('/query', methods=['POST'])
def query():
    start = time.time()
    data = request.json
    question = data.get('query')
    if not question:
        return jsonify({"error": "Query parameter is required"}), 400
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(question, k=3)
    
    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    
    # Initialize OpenAI chat model
    model = ChatOpenAI()

    # Generate response text based on the prompt
    response_text = model.predict(prompt)
    
    # Get sources of the matching documents
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    # Format and return response including generated text and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    end = time.time()
    
    return jsonify({
        "question": question,
        "formatted_response": formatted_response,
        "execution_time": end - start,
        "response_text": response_text
    })

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages)
    
def split_text_into_lines(text):
    return text.splitlines()

def compare_texts(text1, text2):
    prompt = f"""Compare the following two texts and identify differences:
    Text 1:
    {text1}

    Text 2:
    {text2}

    Highlight added, removed, and modified sections."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['choices'][0]['message']['content']

def chunk_and_compare(doc1_lines, doc2_lines):
    differences = []
    for line1, line2 in zip(doc1_lines, doc2_lines):
        diff = compare_texts(line1, line2)
        differences.append(diff)
    return differences

def visualize_diff(text1, text2):
    diff = difflib.unified_diff(
        text1.splitlines(),
        text2.splitlines(),
        lineterm='',
    )
    return "\n".join(diff)

def compare_two_documents():
    # Paths to PDF files
    pdf1_path = "old_version.pdf"
    pdf2_path = "new_version.pdf"

    # Step 1: Extract text
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)

    # Step 2: Split into lines
    lines1 = split_text_into_lines(text1)
    lines2 = split_text_into_lines(text2)

    # Step 3: Compare and visualize
    differences = chunk_and_compare(lines1, lines2)

    # Step 4: Print or save the differences
    response = []
    for diff in differences:
        print(diff)
        response.append(diff)
    
    return response