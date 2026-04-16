from openai import OpenAI
import os


def generate_answer(query, docs, chat_history):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # ✅ Step 1: Remove duplicate chunks
    seen = set()
    unique_docs = []

    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    # ✅ Step 2: Build context + sources
    context_parts = []
    sources = []

    for i, doc in enumerate(unique_docs, start=1):
        page = doc.metadata.get("page", "?")
        # Pages are 0-indexed in PyPDFLoader, show as human-readable
        display_page = page + 1 if isinstance(page, int) else page
        snippet = doc.page_content.strip()

        context_parts.append(f"[{i}] (Page {display_page}):\n{snippet}")

        sources.append({
            "ref": i,
            "page": display_page,
            "snippet": snippet[:300]
        })

    context = "\n\n---\n\n".join(context_parts)

    # ✅ Step 3: Chat history (last 6 turns for follow-ups)
    history_text = ""
    for turn in chat_history[-6:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    # ✅ Step 4: Smarter prompt — generous, finds partial info, infers naturally
    system_prompt = """You are an expert document analyst. Your job is to answer questions based on document excerpts provided to you.

Key rules:
1. Read ALL provided context carefully before answering.
2. Extract and synthesize information — even if it appears in list form, table form, or keywords scattered across chunks.
3. If the answer is explicitly in the context, give it directly and cite the source like [1], [2].
4. If the answer can be reasonably inferred from the context (e.g., the question says "author" but the document says "written by" or shows a name on the title), still provide it.
5. If the information is partially available, give the partial answer and note what's missing.
6. Do NOT say "I could not find that" unless you have genuinely searched every chunk and found zero relevant information.
7. Keep answers clear, structured, and concise.
8. Always cite which chunk(s) support your answer using [1], [2], etc."""

    user_prompt = f"""Here are excerpts from the uploaded document:

{context}

Previous conversation:
{history_text}

User's question: {query}

Instructions:
- Read all excerpts above and find any information related to the question.
- Look for names, titles, labels, headings, or any indirect indicators that answer the question.
- Provide a helpful, accurate answer based on what is available in the context.

Answer:"""

    # ✅ Step 5: LLM call
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,      # Lower temp = more factual, less hallucination
        max_tokens=1500,      # Increased from 1000 → allows complete answers
        top_p=0.9,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }