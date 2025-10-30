import os
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangChainPinecone
import streamlit as st 
import time 

load_dotenv()

# Pinecone Configuration
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "shop-product-catalog"

# Connect to the index
my_index=pc.Index(index_name)
time.sleep(1)

# Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

vectorstore = LangChainPinecone(
    index=my_index,
    embedding=model,
    text_key="Description"   
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

system_message=("""
    You are a helpful and respective shop assistant who answers queries relevant only to the products known to you.
    Please answer all the questions as a professional and customer-friendly tone. If a query lacks a direct answer related to the product,
    then generate a response based on related features. If a question is anything other than shopping, reply with, 'I can only provide answers related to the store only.' 
""")

def gen_answer(system_message, chat_history, prompt):
    """
    Generate AI response using Gemini model with memory of chat history.
    """

    # Build conversation messages in proper chat format
    chat = model.start_chat(history=[
            {"role": "user", "parts": [system_message]},
            *[
                {"role": "user" if msg.startswith("User:") else "model",
                 "parts": [msg.split(":", 1)[1].strip()]}
                for msg in chat_history
            ]
        ])

    
    # Generate response
    response = chat.send_message(prompt)
    answer = response.text.strip()

    # Update chat history
    chat_history.append(f"User: {prompt}")
    chat_history.append(f"Assistant: {answer}")

    return answer


def get_relevant_chunk(query, vectorestore):
    results=vectorestore.similarity_search(query, k=1)
    if results:
        metadata=results[0].metadata
        context=(
            f'Product Name: {metadata.get("ProductName","Not Available")}\n'
            f'Brand: {metadata.get("Brand","Not Available")}\n'
            f'Price: {metadata.get("Price","Not Available")}\n'
            f'Color: {metadata.get("Color","Not Available")}\n'
            f'Description: {results[0].page_content}'
            )
        return context
    return "No relevant search"

def make_prompt(query,context):
    return f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"


st.title("Shop Catalog Chatbot")

query=st.text_input("Ask query...")

if st.button("Get Answer"):
    if query:
        relevant_text=get_relevant_chunk(query, vectorstore)
        prompt=make_prompt(query, relevant_text)

        answer=get_answer(system_meessage, st.session_state.chat_history,prompt)
        st.write("Answer: ", answer)

        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(chat)