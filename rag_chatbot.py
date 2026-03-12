import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from mcq_generator import generate_mcq
from voice_tutor import voice_to_text

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="SSC AI Tutor", page_icon="📚")

st.title("📚 SSC AI Tutor")
st.write("Ask questions from your SSC textbooks")

# -------------------------------
# Load Vector Databases
# -------------------------------

@st.cache_resource
def load_all_dbs():

    embeddings = OpenAIEmbeddings()

    civics_db = FAISS.load_local(
        "vector_store/Maharashtra-Board-Class-6-Civics-Textbook-in-English_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    geography_db = FAISS.load_local(
        "vector_store/Maharashtra-Board-Class-6-Geography-Textbook-in-English_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    history_db = FAISS.load_local(
        "vector_store/Maharashtra-Board-Class-6-History-Textbook-in-English_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return civics_db, geography_db, history_db


civics_db, geography_db, history_db = load_all_dbs()

# -------------------------------
# LLM
# -------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -------------------------------
# Chat History
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Voice Button near chat
# -------------------------------

col1, col2 = st.columns([8,1])

with col2:
    if st.button("🎤"):
        voice_query = voice_to_text()

        if voice_query:
            st.session_state.voice_query = voice_query

# -------------------------------
# Chat Input
# -------------------------------

prompt = st.chat_input("Ask a question from textbook")

# If voice input exists, use it
if "voice_query" in st.session_state:
    prompt = st.session_state.voice_query
    del st.session_state.voice_query

# -------------------------------
# Process Question
# -------------------------------

if prompt:

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context from all DBs
    docs1 = civics_db.similarity_search(prompt, k=3)
    docs2 = geography_db.similarity_search(prompt, k=3)
    docs3 = history_db.similarity_search(prompt, k=3)

    docs = docs1 + docs2 + docs3

    context = "\n".join([doc.page_content for doc in docs])

    prompt_template = f"""
Answer the question using the context below.

Context:
{context}

Question:
{prompt}
"""

    response = llm.invoke(prompt_template)
    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

        # -------------------------------
        # Practice MCQ
        # -------------------------------
        st.subheader("📝 Practice MCQ")

        mcq = generate_mcq(context)
        st.write(mcq)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )