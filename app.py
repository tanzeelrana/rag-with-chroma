from flask import Flask, request, jsonify
import os
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
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
import openai
import pdfplumber
from langchain.prompts import PromptTemplate

from openai import OpenAI
client = OpenAI()

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chromaDB")  
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./upload")

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer.
    2. If you find the answer, write a detailed answer in an HTML format and do not skip any information in the answer.
    3. If the question asks about comparing something, make sure to compare everything and also provide a detailed comparison table in HTML format and do not skip any information in the answer.
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
        chunk_size=800,  # Size of each chunk in characters
        chunk_overlap=50,  # Overlap between consecutive chunks
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

@app.route('/upload_documents', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
    
            load_file_into_db(file_path)
    
    return jsonify({"message": "All files uploaded and processed successfully"}), 200

@app.route('/query', methods=['POST'])
def query():
    start = time.time()
    data = request.json
    question = data.get('query')
    relevant_docs = data.get('relevant_docs')
    if not question:
        return jsonify({"error": "Query parameter is required"}), 400
        
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    
    if not relevant_docs or len(relevant_docs) == 0:
        results = db.similarity_search(question, k=50)
    else:
        
        docs = []
        for doc in relevant_docs:
            docs.append({"source": {"$eq": f"./upload/{doc}"}})
        
        if len(docs) > 1:
            condition = {"$or": docs}
        else:
            condition = docs[0]
        
        filter_retriever = VectorStoreRetriever(
            vectorstore=db,
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold":0.01, "filter": condition, "k": 50}
        )
        results = filter_retriever.get_relevant_documents(question)
    
    if len(results) == 0:
        end = time.time()
        return jsonify({
            "question": question,
            "formatted_response": [],
            "execution_time": end - start,
            "response_text": "Unable to find a valid answer in the given documents"
        })
    else:
        # Combine context from matching documents
        context_text = "\n\n - -\n\n".join([doc.page_content for doc in results])
        
        # Create prompt template using context and query text
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=question)
        
        # Initialize OpenAI chat model
        model = ChatOpenAI(max_tokens=4096, temperature=0, stop=["\nSources:"])

        # Generate response text based on the prompt
        response_text = model.predict(prompt)
        
        # Get sources of the matching documents
        sources = [doc.metadata.get("source", None) for doc in results]
        
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

def compare_texts(context):
    prompt = f"""Compare the following texts and identify differences:
    {context}

    Highlight added, removed, and modified sections. Response text should be html format."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}]
    )
    
    return response.choices[0].message.content

def compare_documents(file_paths):
    
    context = ""
    
    for i, file_path in enumerate(file_paths):
        text = extract_text_from_pdf(file_path)
        context += f"""
        
        Text {i}:
        {text}
        
        """
    
    differences = compare_texts(context)
    return differences

@app.route('/compare', methods=['POST'])
def compare():
    file_paths = []
    
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400
    
    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
    
            file_paths.append(file_path)
    
    differences = compare_documents(file_paths)
    
    return jsonify({"differences": differences}), 200

