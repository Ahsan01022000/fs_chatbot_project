Full Stack Web ChatBot ReadMe
Overview
This project implements a ChatBot for the Full Stack Academy website using LangChain, FAISS, and Google's Gemini LLM (gemini-1.5-flash). The chatbot provides context-aware answers to user queries based on the content of the Full Stack Academy website.

Features
Dynamic Question Answering: Users can ask questions related to the Full Stack Academy website, and the chatbot will provide concise answers.
Unstructured Data Handling: Content is dynamically fetched from the specified URL.
Data Splitting and Vectorization: The content is processed and converted into vector embeddings for efficient retrieval.
FAISS for Fast Retrieval: FAISS is used for efficient similarity search and retrieval of relevant content.
LLM Integration: Google's Gemini LLM is used to generate context-based answers.
Streamlit UI: A user-friendly interface for querying and interacting with the chatbot.
Setup Instructions
Prerequisites
Python 3.7 or higher
Streamlit
LangChain
FAISS
Hugging Face Transformers
Google Gemini API Key
Installation
Clone the Repository:
git clone https://github.com/Ahsan01022000/fullstack-web-chatbot.git
cd fullstack-web-chatbot

Create a Virtual Environment:
python -m venv myenv
source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`

Install Dependencies:
pip install -r requirements.txt

Running the Application:
Set Up the API Key: Ensure you have your Gemini API key.

Run the Streamlit App:
streamlit run app.py

Interact with the ChatBot:
Enter your Gemini API key in the sidebar.
Ask any question related to the Full Stack Academy website.
View the generated answer.

Code Explanation
1. Streamlit Setup
Title and Sidebar: The app has a title and a sidebar for the Gemini API key input.
Question Input: Users can input their questions in a text field.
2. Data Loading
UnstructuredURLLoader: Fetches content from the Full Stack Academy website.
3. Text Splitting
RecursiveCharacterTextSplitter: Splits the fetched content into chunks of 1000 characters with an overlap of 200 characters for better context retention.
4. Embeddings
HuggingFaceEmbeddings: Converts text chunks into dense vector embeddings using the sentence-transformers/all-distilroberta-v1 model.
5. Vector Database
FAISS: Stores the vectorized chunks for efficient similarity search.
6. LLM Integration
ChatGoogleGenerativeAI: Initializes Google's Gemini LLM with parameters such as temperature and maximum token output.
7. Prompt Template
PromptTemplate: Defines the template for how the LLM should answer questions using the retrieved context.
8. Retrieval QA Chain
RetrievalQA: Combines the retriever and the LLM to generate answers based on the retrieved context.
Customization
URL: Change the URL in the URLs list to fetch content from a different website.
Embedding Model: Replace sentence-transformers/all-distilroberta-v1 with another model if needed.
LLM Parameters: Adjust parameters such as temperature and max_output_tokens for different answer styles.
Prompt Template: Modify QA_CHAIN_PROMPT to change the response style.
Troubleshooting
API Key Errors: Ensure the Gemini API key is correct and has necessary permissions.
Module Errors: Check if all dependencies are correctly installed and the virtual environment is activated.
File Not Found: Ensure all required files are in the correct directory
