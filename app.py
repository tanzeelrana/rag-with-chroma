from flask import Flask, request, jsonify
import os
import shutil
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from huggingface_hub import login
import time
from langchain_community.document_loaders import PyPDFLoader
from werkzeug.utils import secure_filename
from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain_community.chat_models import ChatOpenAI # Import OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chromaDB")  
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./upload")

PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.
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

def initialize_retrieval_qa():
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.

    {context}

    Question: {question}

    Helpful Answer:
    """
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=LLM,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )

@app.route('/query', methods=['POST'])
def query():
    start = time.time()
    data = request.json
    question = data.get('query')
    if not question:
        return jsonify({"error": "Query parameter is required"}), 400
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(question, k=3)
    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.1:
        print(f"Unable to find matching results.")
    
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

