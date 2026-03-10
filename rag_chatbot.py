import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load vector database
@st.cache_resource
def load_db():
    embeddings = OpenAIEmbeddings()
    
    db = FAISS.load_local(
        "vector_store/Maharashtra-Board-Class-6-Civics-Textbook-in-English_db", 
        embeddings,
        allow_dangerous_deserialization = True      

    )
    
    return db
db = load_db()

#LLM for generating answer
llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0,
)

st.title("📚 SSC AI Tutor")
st.write("Ask question from your textbook")

query = st.text_input("Ask your question: ")

if query:
    docs = db.similarity_search(query, k=3)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Answer the question using the textbook context below.
    
    context:
    {context}
    
    question:
    {query}
    
    """
    
    response = llm.invoke(prompt)    

    st.subheader("AI Answer")
    st.write(response.content)
    