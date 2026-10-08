from rag import ask, load_llm, load_vectorstore

db = load_vectorstore()
llm = load_llm()

user_query = input("Write Query Here: ")
answer, docs = ask(user_query, db, llm)

print("RESULT:", answer)
print("SOURCES:")
for doc in docs:
    print("-", doc.metadata.get("source"), "page", doc.metadata.get("page", 0) + 1)
