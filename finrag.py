import os

import streamlit as st

from rag import ask, load_llm, load_vectorstore


# Lets the app read the key from Streamlit Cloud secrets when deployed
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


@st.cache_resource
def get_llm():
    return load_llm()


def format_sources(docs):
    """Create a clean source list from retrieved documents."""

    sources = []
    seen = set()

    for doc in docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown document")
        source = os.path.basename(source)

        page = metadata.get("page")

        if page is not None:
            page = int(page) + 1
            source_text = f"📄 **{source}** — Page {page}"
        else:
            source_text = f"📄 **{source}**"

        if source_text not in seen:
            seen.add(source_text)
            sources.append(source_text)

    if not sources:
        return "No source information available."

    return "\n\n".join(sources)


def main():
    st.title("finRAG")

    st.caption(
        "Ask questions from your finance documents. "
        "General information only, not financial advice."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 Sources"):
                    st.markdown(message["sources"])

    prompt = st.chat_input("Ask a finance question")

    if prompt:

        # Display user question
        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        try:

            # Run RAG
            answer, docs = ask(
                prompt,
                get_vectorstore(),
                get_llm()
            )

            # Format retrieved sources
            sources = format_sources(docs)

            # Display answer
            with st.chat_message("assistant"):

                st.markdown(answer)

                with st.expander("📚 Sources"):
                    st.markdown(sources)

            # Save conversation
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except FileNotFoundError:

            st.error(
                "Vector store not found. "
                "Run create_memory_for_llm.py first."
            )

        except KeyError:

            st.error(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()