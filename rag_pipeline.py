import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings()

textbook_path = "textbooks"
vector_path = "vector_store"

for file in os.listdir(textbook_path):

    if file.endswith(".pdf"):

        subject = file.replace(".pdf","")

        loader = PyPDFLoader(os.path.join(textbook_path,file))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, embeddings)

        save_path = os.path.join(vector_path, subject+"_db")

        vector_db.save_local(save_path)

        print(subject,"vector DB created")
        
db = FAISS.load_local(
    "vector_store/ssc-ai-platform\vector_store\Maharashtra-Board-Class-6-Civics-Textbook-in-English_db",
    embeddings,
    allow_dangerous_deserialization=True
)

docs = db.similarity_search("What is photosynthesis?")

print(docs[0].page_content)
