import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

def main():
    st.header("chat with pdf")
    load_dotenv()
    pdf = st.file_uploader("upload your pdf", type = 'pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text =""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =1000,
            chunk_overlap=200,
            length_function=len
        )
        chuncks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)    
            st.write('embeddings loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chuncks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)

            st.write('embeddings computation completed')
        query = st.text_input("ask questions about ur pdf file:")
        st.write(query)
        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm,chain_type ="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()