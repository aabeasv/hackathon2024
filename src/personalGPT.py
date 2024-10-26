import os
import time

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
# from langchain_community.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# The LLM Model
model = os.environ.get("MODEL", "mistral")
# Embeddings model
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")

persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))


def main():
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Specify the Chroma db
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)
    # LangChain QA
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=False)
    # Questions and answers
    while True:
        # Prompt for a query
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], []
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(answer)


if __name__ == "__main__":
    main()
