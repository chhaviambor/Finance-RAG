import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

qa_chain = None
app_state = {"status": "uninitialized"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain, app_state
    print("Initializing Free Groq RAG system...")
    
    pdf_path = "data/fomcminutes20240918.pdf"
    if not os.path.exists(pdf_path):
        print(f"Warning: Data file not found at {pdf_path}")
        app_state["status"] = "missing_pdf"
    else:
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            
            # Vector DB Chunking Strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)
            
            # LOCAL Free Embeddings (Cost & Server Size Optimization)
            print("Loading ultra-lightweight FastEmbed models...")
            embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            vectorstore = FAISS.from_documents(splits, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # API Setup with Groq (Instant & Free execution)
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                app_state["status"] = "missing_api_key"
            else:
                llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=150)
                
                template = (
                    "You are an expert financial assistant strictly answering questions based on the provided FOMC minutes.\n\n"
                    "CRITICAL GUARDRAILS:\n"
                    "1. If the user's question asks about topics outside of finance, economics, or the FOMC, you MUST reject it by answering EXACTLY: 'This question is outside the scope of this system.'\n"
                    "2. If you cannot find the answer to the user's question within the Context provided below, you MUST answer EXACTLY: 'I do not have enough information to answer this.'\n"
                    "3. Do not hallucinate or use any outside knowledge whatsoever. Base your answer purely on the Context.\n"
                    "4. Be concise and minimize verbosity.\n\n"
                    "Context:\n"
                    "{context}\n\n"
                    "Question: {question}\n"
                    "Helpful Answer:"
                )
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                def format_docs(documents):
                    return "\n\n".join(doc.page_content for doc in documents)
                    
                qa_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                app_state["status"] = "ready"
                print("RAG system successfully initialized.")
        except Exception as e:
            print(f"Failed to initialize RAG system: {e}")
            app_state["status"] = "initialization_failed"
            
    yield
    print("Shutting down...")

app = FastAPI(title="Finance RAG System - FOMC Assistant", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": app_state["status"]}

@app.post("/query", response_model=QueryResponse)
def query_fomc(request: QueryRequest):
    if not request.question or not request.question.strip():
        return QueryResponse(answer="Please provide a valid question.")
        
    if app_state["status"] == "missing_pdf":
        return QueryResponse(answer="Knowledge base not found.")
        
    if app_state["status"] == "missing_api_key":
        return QueryResponse(answer="The system requires a Groq API key to generate answers. Please configure your GROQ_API_KEY environment variable.")
        
    if qa_chain is None:
        return QueryResponse(answer="The RAG system is currently unavailable due to an initialization error.")
    
    try:
        answer = qa_chain.invoke(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        # Proper error handling for model failure / API quota issues
        error_msg = str(e)
        print(f"Model generation error: {error_msg}")
        if "401" in error_msg or "authentication" in error_msg.lower():
            return QueryResponse(answer="Authentication Error: Your GROQ API Key is invalid or expired. Please check your .env file.")
        
        return QueryResponse(answer=f"AI Provider Error: {error_msg}")
