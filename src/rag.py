import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "insurance_documents"


# --------------------------------------------------
# 1. Embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)


# --------------------------------------------------
# 3. Create retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# 4. Gemini LLM
# --------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=api_key,
    max_retries=3,
)


# --------------------------------------------------
# 5. Grounded RAG function
# --------------------------------------------------

def ask_rag(question):

    documents = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"SOURCE: {doc.metadata.get('source')}\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    prompt = f"""
You are an insurance claims assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not use outside knowledge.

If the answer cannot be determined from the context,
say:

"I cannot determine that from the provided documents."

Do not invent policy conditions, claim amounts,
coverage rules, or other facts.

Always provide the source documents used.

CONTEXT
-------
{context}

QUESTION
--------
{question}

ANSWER
------
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        answer = response.content

    else:
        answer_parts = []

        for part in response.content:
            if isinstance(part, dict) and part.get("text"):
                answer_parts.append(part["text"])

        answer = "\n".join(answer_parts)

    return answer, documents


# --------------------------------------------------
# 6. Command-line test
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nEnter your insurance question: "
    )

    answer, documents = ask_rag(question)

    print("\n========== GROUNDED ANSWER ==========\n")
    print(answer)

    print("\n========== SOURCES ==========\n")

    for i, doc in enumerate(documents, start=1):

        print(
            f"{i}. "
            f"{doc.metadata.get('source')}"
        )