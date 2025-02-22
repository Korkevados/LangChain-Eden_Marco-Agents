import os
from dotenv import  load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import  ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import  PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()

if __name__ == "__main__":
    print("Retrevieng")

    query = "what is pinecone in mechine learning"
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)

    retrevial_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm,retrevial_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=combine_docs_chain)

    result = retrival_chain.invoke(input={"input":query})

    print(result['answer'])

