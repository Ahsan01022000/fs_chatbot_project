import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Full Stack web ChatBot")
st.sidebar.header("Please Provide your Gemini key")
api_key=st.sidebar.text_input("Please Enter Your Gemini key",type="password")

st.markdown("Please Ask any of your question related to FULLSTACK ACADEMY WEBSITE")
question=st.text_input("Please Enter your question here: ")

#API key provided
if api_key:
    try:
        #load the url
        URLs=["https://fullstackacademy.in/"]
        loader=UnstructuredURLLoader(urls=URLs)
        data=loader.load()
        

        
        #spliting the data into chunks
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=text_splitter.split_documents(data)
        
        #setting up embedding model
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-distilroberta-v1")
        
        #database which is Faiss
        vectordatabase=FAISS.from_documents(chunks,embedding_model)
        
        #Intialize the llm
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key,temperature=0.7,max_output_tokens=256)
        
        
        #Lets create a prompt template
        template="""Use the context stricty to provide a concise answer. If you dont know the answer just say you dont know
        {context}
        Question:{question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT=PromptTemplate.from_template(template)
        
        #Setting up Retreival QA Chain
        chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",
                                                retriever=vectordatabase.as_retriever(search_kwargs={"k":5}),
                                               chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
        
        if question:
            with st.spinner("Searching......"):
                answer=chain.run(question)
            
            st.write(answer)
            
            
    except Exception as e:
        st.error(f"An error occured: {e}")
        
else:
    st.warning("Please Enter your gemini Key")