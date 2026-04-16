
from openai import OpenAI
import os


def generate_answer(query, docs, chat_history):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    context_parts = []
    sources = []

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip()

        context_parts.append(f"[{i}] (Page {page}): {snippet}")

        sources.append({
            "ref": i,
            "page": page,
            "snippet": snippet[:200]
        })

    context = "\n\n".join(context_parts)

    history_text = ""
    for turn in chat_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    system_prompt = """
You are a helpful assistant.

Rules:
- Answer ONLY from provided context
- Always cite like [1], [2]
- Use chat history for follow-ups
- If answer not found say: "I could not find that in the document."
"""

    user_prompt = f"""
Previous conversation:
{history_text}

Context:
{context}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }