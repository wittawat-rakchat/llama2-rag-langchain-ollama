import argparse
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains import RetrievalQA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        type=str,
        default='http://example.com',
        required=True,
        help='The URL to filter out.'
    )
    arg_parser = parser.parse_args()
    url = arg_parser.url
    print(f'Get the information from {url} to summarise.')

    loader = WebBaseLoader(url)
    data = loader.load()

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_text = text_splitter.split_documents(data)

    # vector store
    vector_store = Chroma.from_documents(
        documents=split_text,
        embedding=GPT4AllEmbeddings()
    )

    # load LLM
    llm = Ollama(
        model='llama2',
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    print(f'+++++ Using {llm.model} model +++++')

    # get RAG prompt
    rag_prompt = hub.pull('rlm/rag-prompt-llama')

    # create Q & A chain
    retrieval_qa = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt}
    )

    # Ask a question
    question = f'What are the latest news on {url}?'
    result = retrieval_qa({"query": question})

    # print(result)

if __name__ == "__main__":
    main()







