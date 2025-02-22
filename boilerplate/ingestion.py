import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import  OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Hello")
    loader = TextLoader("C:\\Udemy\\LangChain\\boilerplate\\vectordatabase.txt", encoding="utf-8")
    document = loader.load()
    print("Succesful loaded")

    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"created {len(texts)} chunks")
    # print(f"{os.environ.get("OPENAI_API_KEY"}")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    print("create index")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])

    print("finished")








