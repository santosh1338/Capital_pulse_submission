import streamlit as st
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_llm_and_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, embeddings

@st.cache_resource
def build_vector_store(_df, _embeddings):
    df = _df.copy()
    df["ma"] = df["Close"].rolling(5).mean()
    df["trend"] = np.where(df["Close"] > df["ma"], "Uptrend", "Downtrend")
    
    docs = []
    for date, row in df.tail(100).iterrows():
        content = (f"Date: {date.date()}. Close: {row['Close']:.2f}. "
                   f"Trend: {row['trend']}. Volume: {row['Volume']}.")
        docs.append(Document(page_content=content))
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    
    vectorstore = SKLearnVectorStore.from_documents(documents=split_docs, embedding=_embeddings)
    return vectorstore

def render_chatbot(df):
    st.subheader("Market Analyst Chatbot")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.spinner("Initializing AI Engine..."):
        llm, embeddings = load_llm_and_embeddings()
        vectorstore = build_vector_store(df, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about market trends..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.spinner("Thinking..."):
            try:
                docs = retriever.invoke(user_input)
                context_text = "\n".join([d.page_content for d in docs])
                
                prompt = f"Context:\n{context_text}\n\nQuestion: {user_input}\n\nAnswer succinctly:"
                response = llm.invoke(prompt)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")