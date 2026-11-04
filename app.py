
import streamlit as st
import fitz
import numpy as np
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="SCSVMV AcademicQuery AI",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "logo.jpg"

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fc;
}

[data-testid="stSidebar"] {
    background-color: #102b50;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    color: #102b50;
    font-size: 36px;
    font-weight: 800;
}

.subtitle {
    color: #64748b;
    font-size: 17px;
}

.hero {
    background: linear-gradient(120deg, #102b50, #24558a);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-top: 20px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 32px;
}

.hero p {
    color: #e5edf8;
    font-size: 17px;
}

.info-card {
    background-color: white;
    padding: 24px;
    border-radius: 15px;
    border: 1px solid #e1e8f0;
    min-height: 160px;
    box-shadow: 0 4px 15px rgba(16,43,80,0.06);
}

.info-card h3 {
    color: #102b50;
}

.info-card p {
    color: #64748b;
}

.stButton button {
    background-color: #173f70;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
}

.stButton button:hover {
    background-color: #d5a943;
    color: #102b50;
}

[data-testid="stChatMessage"] {
    background-color: white;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def split_text(text, chunk_size=120, overlap=25):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def initialize_session():
    if "records" not in st.session_state:
        st.session_state.records = []

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def process_documents(uploaded_files):
    model = load_model()

    new_records = []
    processed_count = 0

    progress = st.progress(0)

    for file_index, uploaded_file in enumerate(uploaded_files):

        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash in st.session_state.processed_files:
            continue

        try:
            pdf = fitz.open(
                stream=file_bytes,
                filetype="pdf"
            )

            for page_number, page in enumerate(pdf, start=1):

                page_text = page.get_text("text")

                if not page_text.strip():
                    continue

                chunks = split_text(page_text)

                for chunk in chunks:
                    new_records.append({
                        "file": uploaded_file.name,
                        "page": page_number,
                        "text": chunk
                    })

            st.session_state.processed_files.add(file_hash)
            processed_count += 1

        except Exception as error:
            st.error(
                f"Error processing {uploaded_file.name}: {error}"
            )

        progress.progress(
            (file_index + 1) / len(uploaded_files)
        )

    if new_records:
        with st.spinner("Creating document embeddings..."):

            texts = [record["text"] for record in new_records]

            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for record, embedding in zip(new_records, embeddings):
                record["embedding"] = embedding

            st.session_state.records.extend(new_records)

    progress.empty()

    return processed_count


def get_answer(question):
    if not st.session_state.records:
        return []

    model = load_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = []

    for record in st.session_state.records:
        score = float(
            np.dot(question_embedding, record["embedding"])
        )

        scores.append((score, record))

    scores.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scores[:3]


initialize_session()


st.sidebar.markdown(
    "<h2 style='text-align:center; color:white;'>SCSVMV</h2>",
    unsafe_allow_html=True
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=140)
else:
    st.sidebar.markdown(
        "<h1 style='text-align:center;'>🎓</h1>",
        unsafe_allow_html=True
    )
    st.sidebar.warning("Place logo.png inside the assets folder.")

st.sidebar.markdown(
    """
    <p style='text-align:center; color:#d5a943;'>
    AcademicQuery AI
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Student Chatbot",
        "Admin Upload",
        "About Project"
    ]
)


if page == "Home":

    st.markdown(
        '<div class="main-title">SCSVMV Academic Portal</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Your Smart University Information Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h1>Welcome to AcademicQuery AI</h1>
        <p>
        Find university academic information quickly.
        Ask questions about attendance, syllabus,
        examinations and university regulations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore Academic Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎓 Student Support</h3>
            <p>Find academic information and university guidelines.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI Chatbot</h3>
            <p>Ask questions in simple English and find answers.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📚 Digital Library</h3>
            <p>Search university regulations and academic PDFs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Academic Information Dashboard")

    col1, col2 = st.columns(2)

    document_count = len(set(
        record["file"] for record in st.session_state.records
    ))

    with col1:
        st.metric("Documents Processed", document_count)

    with col2:
        st.metric(
            "Text Sections",
            len(st.session_state.records)
        )

    st.info(
        "Select Student Chatbot from the sidebar "
        "to ask your academic questions."
    )


elif page == "Student Chatbot":

    st.title("🤖 Student Academic Chatbot")

    st.write(
        "Ask questions about university academic documents."
    )

    if not st.session_state.records:
        st.warning(
            "No documents are available. "
            "Please upload PDF documents from Admin Upload."
        )

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message.get("sources"):
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"File: {source['file']} | "
                            f"Page: {source['page']}"
                        )
                        st.write(source["text"])

    question = st.chat_input("Type your academic question here...")

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            if not st.session_state.records:
                answer = "Please upload academic PDF documents first."
                st.warning(answer)

                sources = []

            else:
                try:
                    with st.spinner("Searching academic documents..."):

                        results = get_answer(question)

                    if results and results[0][0] > 0.20:

                        best_score, best_record = results[0]

                        answer = (
                            "I found the following relevant information "
                            "in your university documents:\n\n"
                            + best_record["text"]
                        )

                        st.write(answer)

                        sources = [
                            record for score, record in results
                            if score > 0.20
                        ]

                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(
                                    f"**File:** {source['file']}"
                                )
                                st.write(
                                    f"**Page:** {source['page']}"
                                )
                                st.write(source["text"])
                                st.divider()

                    else:
                        answer = (
                            "Sorry, I could not find relevant information "
                            "in the uploaded documents."
                        )

                        st.write(answer)
                        sources = []

                except Exception as error:
                    answer = f"An error occurred: {error}"
                    st.error(answer)
                    sources = []

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })


elif page == "Admin Upload":

    st.title("📚 Admin Document Upload")

    st.write(
        "Upload university academic PDF documents "
        "to make them searchable."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):

        if uploaded_files:

            try:
                with st.spinner("Processing your documents..."):

                    count = process_documents(uploaded_files)

                st.success(
                    f"Processing completed. "
                    f"{count} new document(s) processed."
                )

            except Exception as error:
                st.error(f"Processing failed: {error}")

        else:
            st.warning("Please select at least one PDF file.")

    st.divider()

    st.subheader("Uploaded Documents")

    document_names = sorted(set(
        record["file"] for record in st.session_state.records
    ))

    if document_names:

        for name in document_names:
            st.write("📄", name)

    else:
        st.info("No documents have been processed yet.")

    if st.button("Clear All Processed Documents"):

        st.session_state.records = []
        st.session_state.processed_files = set()
        st.session_state.chat_history = []

        st.success("Processed documents and chat history cleared.")
        st.rerun()


elif page == "About Project":

    st.title("About AcademicQuery AI")

    st.write("""
    AcademicQuery AI is an NLP-based university academic
    question-answering system.

    The system helps students find information from
    uploaded academic PDF documents.
    """)

    st.subheader("Project Objectives")

    st.write("1. To provide quick academic information.")
    st.write("2. To reduce manual searching of PDF documents.")
    st.write("3. To use Natural Language Processing.")
    st.write("4. To provide relevant answers with document sources.")

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Natural Language Processing
    - Sentence Transformers
    - PyMuPDF
    - NumPy
    """)

    st.subheader("System Architecture")

    st.write("""
    Student
       ↓
    Streamlit Website
       ↓
    Student Question
       ↓
    NLP Processing
       ↓
    Semantic Search
       ↓
    Academic PDF Documents
       ↓
    Relevant Answer with Source
    """)

    st.info(
        "This is a student project prototype. "
        "Use verified university documents for accurate information."
    )
