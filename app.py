
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from IPython.display import Markdown as md
import streamlit as st


st.title("Question and Answers using Retrieval-Augmented Generation System")

# Setup Google API Key
GOOGLE_API_KEY = "api_dumykeys"

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

# Initialize embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="D:/lang chain _rag/lang chain_genai/chroma_db_/chroma_db_/chroma.sqlite3_./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from the user. Your answer should be based on the specific context."),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context./n/nContext: {context}/n/nQuestion: {question}/n/nAnswer: ")
])

output_parser = StrOutputParser()

from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

user_question = st.text_input("Enter your question here:")
if st.button("Ask"):
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)
    st.write(md(response))


