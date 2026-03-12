import os
#from urllib import response
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI



load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# embeddings = OpenAIEmbeddings()

# db = FAISS.load_local(
#     "vector_store/Maharashtra-Board-Class-6-Civics-Textbook-in-English_db",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# docs = db.similarity_search("What is Society?")

# print(docs[0].page_content)

def ask_question(query):

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        "vector_store/Maharashtra-Board-Class-6-Civics-Textbook-in-English_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve relevant document
    docs = db.similarity_search(query, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    # Get response
    response = llm.invoke(prompt)

    return response.content


# Print answer
print(ask_question("What is Society?"))