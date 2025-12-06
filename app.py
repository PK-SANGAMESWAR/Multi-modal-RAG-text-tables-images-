import streamlit as st
import json
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# ---- UTILS ----

def partition_document(file_path):
    return partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=False,   # avoid transformer crash unless installed
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )


def create_chunks(elements):
    return chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )


def separate_content_types(chunk):
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
    }

    if hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            category = getattr(element, "category", None)

            if category == "Table":
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            if category == "Image":
                img = getattr(element.metadata, "image_base64", None)
                if img:
                    content_data["images"].append(img)

    return content_data


def create_ai_enhanced_summary(text, tables, images):
    llm = ChatOllama(model="llama3.2:3b")

    prompt = "You are generating a searchable semantic summary.\n\n"
    prompt += f"TEXT CONTENT:\n{text}\n\n"

    if tables:
        prompt += "TABLES:\n"
        for i, t in enumerate(tables):
            prompt += f"[TABLE {i+1}]\n{t}\n\n"

    if images:
        prompt += f"[{len(images)} images present but omitted]\n\n"

    prompt += """TASK:
Write a detailed, information-rich summary with key findings, numbers, insights, and themes.
Return ONLY the summary.
"""

    response = llm.invoke(prompt)
    return response.content


def summarise_chunks(chunks):
    docs = []

    for chunk in chunks:
        c = separate_content_types(chunk)
        if c["tables"] or c["images"]:
            enhanced = create_ai_enhanced_summary(c["text"], c["tables"], c["images"])
        else:
            enhanced = c["text"]

        docs.append(
            Document(
                page_content=enhanced,
                metadata={
                    "raw_text": c["text"],
                    "tables_html": json.dumps(c["tables"]),
                    "images_base64": json.dumps(c["images"]),
                }
            )
        )

    return docs


def create_vector_store(documents, persist_directory="db/chroma_db"):
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)

    embedding = OllamaEmbeddings(model="mxbai-embed-large")

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name="rag_collection",
    )
    return db


# ---- STREAMLIT UI ----

st.title("📘 Medical RAG System — PDF AI Analysis")

uploaded_file = st.file_uploader("Upload a medical research PDF", type=["pdf"])

if uploaded_file:
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("📄 Extracting PDF elements...")
    elements = partition_document(temp_path)

    st.success(f"Extracted {len(elements)} elements")

    st.info("🔍 Creating content chunks...")
    chunks = create_chunks(elements)
    st.success(f"{len(chunks)} chunks created")

    st.info("🤖 Generating AI summaries for each chunk...")
    docs = summarise_chunks(chunks)

    st.success("Summaries generated!")

    st.info("📦 Creating vector DB...")
    db = create_vector_store(docs)
    st.success("Vector DB ready!")

    query = st.text_input("Ask a question from the PDF")

    if query:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        matches = retriever.invoke(query)

        st.subheader("🔎 Top Relevant Chunks")
        for m in matches:
            with st.expander("Chunk"):
                st.write(m.page_content)
                st.json(m.metadata)

        # Generate final answer
        llm = ChatOllama(model="llama3.2:3b")
        answer_prompt = f"""
You are an expert medical research assistant.

Here is the user question:
{query}

Here are the retrieved relevant chunks:
{[m.page_content for m in matches]}

Write a clear, accurate answer with citations like [Chunk 1], [Chunk 2].
"""

        answer = llm.invoke(answer_prompt)

        st.subheader("🧠 Final AI Answer")
        st.write(answer.content)
