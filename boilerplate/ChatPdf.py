import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import  hub
from dotenv import load_dotenv

load_dotenv()

import textwrap

if __name__ == '__main__':

    pdf_path = "C:\\Udemy\\LangChain\\boilerplate\\2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_reAct")

    new_vectorstore = FAISS.load_local("faiss_index_reAct", embeddings, allow_dangerous_deserialization=True)

    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(OpenAI(), retrival_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(new_vectorstore.as_retriever(), combine_docs_chain)


    print("Hey! Ask me anything about the document. Type 'bye' to exit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Goodbye!")
            break

        res = retrival_chain.invoke({"input": user_input})
        formatted_answer = textwrap.fill(res["answer"], width=80)
        print("Bot:", formatted_answer)


