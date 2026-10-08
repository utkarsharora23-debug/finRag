
import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

TOP_K = 5
RETRIEVAL_K = 8

PROMPT_TEMPLATE = """

You are a finance assistant. Use only the context below to answer the question.

If the answer is not in the context, say you don't know. Do not make up numbers.

Keep the answer short, clear, and in simple words.

For comparison questions, compare the two topics using only information available
in the context. If information about one side is missing, say which information
is missing instead of guessing.

This is general information, not personal investment advice.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.

"""


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_vectorstore():
    return FAISS.load_local(
        DB_FAISS_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def load_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.0,
        api_key=os.environ["GROQ_API_KEY"],
    )


def build_query_variants(question):
    """
    Create additional search queries for common finance terminology
    and comparison questions.
    """

    expanded_question = question

    replacements = {
        "sip": "Systematic Investment Plan (SIP)",
        "mf": "Mutual Fund (MF)",
        "fd": "Fixed Deposit (FD)",
        "f.d.": "Fixed Deposit (FD)",
        "fy": "Financial Year (FY)",
    }

    for short_form, full_form in replacements.items():
        if short_form in expanded_question.lower():
            expanded_question = expanded_question.replace(
                short_form,
                full_form
            )

    queries = [question, expanded_question]

    q = question.lower()

    # SIP
    if "sip" in q:
        queries.extend([
            "Systematic Investment Plan SIP mutual funds",
            "Systematic Investment Plan how does SIP work",
            "SIP periodic investment mutual fund",
        ])

    # Riskometer
    if "riskometer" in q:
        queries.extend([
            "mutual fund Riskometer risk level",
            "Riskometer Low Moderate High Very High mutual funds",
        ])

    # Mutual funds
    if "mutual fund" in q:
        queries.extend([
            "mutual fund investment scheme investors securities",
            "mutual funds features benefits risks",
        ])

    # Fixed Deposit
    if "fd" in q or "fixed deposit" in q:
        queries.extend([
            "fixed deposit FD investment interest risk",
            "bank fixed deposit features returns",
        ])

    # Revenue
    if "revenue" in q:
        queries.extend([
            "Infosys FY2025 revenue total income",
            "Infosys financial performance FY 2025 revenue",
            "Infosys annual report 2024 2025 revenue",
        ])

    # Profit
    if "profit" in q or "net profit" in q:
        queries.extend([
            "Infosys FY2025 net profit",
            "Infosys financial performance FY2025 profit",
            "Infosys annual report net profit 2025",
        ])

    # Comparison questions
    if (
        "compare" in q
        or "comparison" in q
        or "difference" in q
        or " vs " in q
        or "versus" in q
    ):

        # SIP vs FD
        if "sip" in q and ("fd" in q or "fixed deposit" in q):
            queries.extend([
                "Systematic Investment Plan SIP definition features",
                "SIP regular investment mutual fund",
                "Fixed Deposit FD definition features interest",
                "fixed deposit investment maturity interest",
            ])

        # Mutual fund vs FD
        if "mutual fund" in q and (
            "fd" in q or "fixed deposit" in q
        ):
            queries.extend([
                "mutual fund investment returns risk features",
                "Fixed Deposit FD interest risk maturity",
            ])

    # Remove duplicate queries
    unique_queries = []
    seen = set()

    for query in queries:
        key = query.lower().strip()

        if key not in seen:
            seen.add(key)
            unique_queries.append(query)

    return unique_queries


def retrieve_documents(question, db):
    """
    Search FAISS using multiple query variants.
    Keep the best-scoring unique chunks.
    """

    queries = build_query_variants(question)

    best_documents = {}

    for query in queries:

        results = db.similarity_search_with_score(
            query,
            k=RETRIEVAL_K
        )

        for doc, score in results:

            content_key = doc.page_content.strip()

            if (
                content_key not in best_documents
                or score < best_documents[content_key][1]
            ):
                best_documents[content_key] = (doc, score)

    # Lower FAISS distance = better similarity
    ranked_documents = sorted(
        best_documents.values(),
        key=lambda x: x[1]
    )

    return [
        doc
        for doc, score in ranked_documents[:TOP_K]
    ]


def ask(question, db, llm):

    docs = retrieve_documents(
        question,
        db
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    answer = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )

    return answer.content, docs