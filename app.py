import os
import uuid
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pdfplumber
from docx import Document
import io

TEXT_DIR = "texts"
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 30
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")  # Set this in .streamlit/secrets.toml or manually
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

def load_and_chunk_texts(text_dir, chunk_size, overlap):
    chunks = []
    for fname in os.listdir(text_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
                words = f.read().split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    if chunk:
                        chunks.append({
                            "text": chunk,
                            "source": fname
                        })
    return chunks

def setup_qdrant_collection(collection_name, dim):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
    )

def embed_and_store_chunks(chunks, collection_name):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={"text": chunks[i]["text"], "source": chunks[i]["source"]}
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)

def search_documents(query, k=5):
    query_vector = embedder.encode(query, normalize_embeddings=True)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    return results

def build_prompt_from_techniques(query, chunks, selected_techniques):
    context = "\n\n".join([f"{i+1}. {c.payload['text']}" for i, c in enumerate(chunks)])

    # List of techniques with corresponding prompt instructions
    prompt_techniques = [
        "Use chain-of-thought reasoning to break down the problem.",
        "Apply tree-of-thought exploration to consider multiple paths before concluding.",
        "Assume a relevant role and answer from that perspective.",
        "Follow a ReAct style: reason, then act based on context.",
        "Use directional stimulus prompting to guide your thinking with hints from the context.",
        "Step back and re-evaluate your initial assumptions before answering.",
        "Answer with four multiple-choice options (A, B, C, D), then indicate the correct one."
    ]

    if not any(selected_techniques):
        return query

    selected_instructions = [prompt_techniques[i] for i, val in enumerate(selected_techniques) if val]

    instructions = " ".join(selected_instructions)

    prompt = f"""
Use the following context to answer the question. {instructions}

### Context:
{context}

### Question:
{query}

### Let's think step-by-step:
"""
    if selected_techniques[6]:
            prompt += """Make 10 mcq questions and for each give four answer options (A, B, C, D) and indicate the correct one at the end.

Use this format:

Question: [question]

A. option 1  
B. option 2  
C. option 3  
D. option 4  
**Correct Answer:** <A/B/C/D>
"""
    return prompt



def ask_groq_llm(prompt, model=GROQ_MODEL, key=GROQ_API_KEY):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a technical expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def convert_pdf_to_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

def convert_docx_to_text(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

# Streamlit UI

st.set_page_config(page_title="Retrieval-Augmented Generation via Vector Search + LLM", page_icon="🧠", layout="wide")

st.title("Tag based question-answering with LLM Reasoning")
st.markdown("This app uses Qdrant for vector search and Groq's LLaMA with advanced prompt engineering.")

if st.button("🔄 Build Vector DB"):
    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith(".txt")]

    if not text_files:
        st.warning("⚠️ No files selected")
    else:
        with st.spinner("Reading and embedding documents..."):
            chunks = load_and_chunk_texts(TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
            setup_qdrant_collection(COLLECTION_NAME, 384)
            embed_and_store_chunks(chunks, COLLECTION_NAME)

        st.success("✅ Vector DB built")

technique_labels = [
    "Chain-of-Thoughts",
    "Tree-of-Thoughts",
    "Role-based prompting",
    "ReAct prompting",
    "Directional Stimulus prompting",
    "Step-Back prompting",
    "MCQ prompting"
]

query = st.text_input("🔍 Ask a question about the information present in the document files:")

selected_techniques = []
st.markdown("### 🧠 Select Prompting Techniques:")
for label in technique_labels:
    selected = st.checkbox(label)
    selected_techniques.append(1 if selected else 0)
    
if query:
    with st.spinner("Searching and querying LLM..."):
        results = search_documents(query)
        prompt = build_prompt_from_techniques(query, results, selected_techniques)
        try:
            answer = ask_groq_llm(prompt)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # Get selected technique names
    selected_labels = [label for label, sel in zip(technique_labels, selected_techniques) if sel]
    
    if selected_labels:
        st.subheader("🧠 Selected Prompting Techniques:")
        st.markdown(" - " + "\n - ".join(selected_labels))

    st.markdown(answer)
    st.subheader("📄 Sources Used")
    for r in results:
        st.markdown(f"- **{r.payload['source']}** (score: `{r.score:.4f}`)")


# Optional: Upload file interface
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .txt, .pdf, or .docx files", 
    accept_multiple_files=True, 
    type=["txt", "pdf", "docx"]
)
if uploaded_files:
    os.makedirs(TEXT_DIR, exist_ok=True)

    for file in uploaded_files:
        filename = os.path.splitext(file.name)[0]
        extension = os.path.splitext(file.name)[1].lower()

        try:
            if extension == ".txt":
                content = file.read().decode("utf-8")
            elif extension == ".pdf":
                content = convert_pdf_to_text(file)
            elif extension == ".docx":
                content = convert_docx_to_text(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            with open(os.path.join(TEXT_DIR, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write(content)

            st.sidebar.success(f"{file.name} converted and saved.")
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {e}")

    st.sidebar.info("Click 'Build Vector DB' to include uploaded files.")