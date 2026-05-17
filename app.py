import streamlit as st
from src.loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_pipeline import generate_answer
from src.quiz_generator import generate_quiz, parse_quiz_output

st.set_page_config(page_title="AI Student Assistant", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #eef2ff 0%, #ffffff 100%); }
    .stButton>button { background-color: #4b7bec; color: white; border-radius: 8px; border: none; }
    .stButton>button:hover { background-color: #3867d6; }
    .stTextInput>div>div>input { border-radius: 12px; border: 1px solid #d0d7ff; padding: 10px; }
    .stFileUploader>div { border-radius: 16px; border: 2px dashed #a3b1ff; background: #f5f7ff; }
    .css-1d391kg { box-shadow: 0 10px 30px rgba(15, 23, 70, 0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 AI Student Assistant")

# Initialize session state
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
    st.session_state.uploaded_file_name = None
    st.session_state.page = "home"  # home, questions, quiz
    st.session_state.query_text = ""
    st.session_state.answer = None
    st.session_state.sources = []
    st.session_state.quiz_text = None
    st.session_state.quiz_data = []
    st.session_state.quiz_warning = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = []
    st.session_state.chunks = None
    st.session_state.retriever = None
    st.session_state.embeddings = None
    st.session_state.llm = None


def reset_quiz_state():
    st.session_state.quiz_text = None
    st.session_state.quiz_data = []
    st.session_state.quiz_warning = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = []


def build_quiz_context(chunks, max_total_chars=600, max_chunks=1, max_chars_per_chunk=600):
    """Build minimal context for quiz to avoid 413 errors."""
    if not chunks:
        return ""
    text = chunks[0].page_content.strip()[:max_chars_per_chunk]
    return text


# SIDEBAR: PDF Upload
with st.sidebar:
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        if not st.session_state.pdf_loaded or st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with open("data/temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                docs = load_pdf("data/temp.pdf")
                chunks = split_documents(docs)

                embeddings = get_embeddings()
                db = create_vector_store(chunks, embeddings)
                retriever = get_retriever(db, search_k=20)
                llm = get_llm()

                st.session_state.pdf_loaded = True
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chunks = chunks
                st.session_state.retriever = retriever
                st.session_state.embeddings = embeddings
                st.session_state.llm = llm
                st.session_state.page = "home"
                st.session_state.answer = None
                st.session_state.sources = []
                reset_quiz_state()
            st.success(f"✓ Loaded: {uploaded_file.name}")

    if st.session_state.pdf_loaded:
        st.info(f"**Current PDF:** {st.session_state.uploaded_file_name}")


# PAGE NAVIGATION
if st.session_state.pdf_loaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
    with col2:
        if st.button("❓ Ask Question", use_container_width=True):
            st.session_state.page = "questions"
    with col3:
        if st.button("📝 Take Quiz", use_container_width=True):
            st.session_state.page = "quiz"
    st.markdown("---")


# ============ HOME PAGE ============
if not st.session_state.pdf_loaded:
    st.markdown(
        "### Welcome to AI Student Assistant 👋\n\n"
        "1. **Upload a PDF** in the sidebar\n"
        "2. **Ask Questions** - Get instant answers from your document\n"
        "3. **Take a Quiz** - Test your knowledge with AI-generated questions\n\n"
        "Get started by uploading a PDF file!"
    )
elif st.session_state.page == "home":
    st.markdown(f"### Document Loaded: {st.session_state.uploaded_file_name}")
    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            "**Choose what you'd like to do:**\n\n"
            "- **❓ Ask Question** - Ask the PDF anything and get AI-powered answers\n"
            "- **📝 Take Quiz** - Generate a quiz to test your understanding"
        )
    with right:
        st.info(
            "**Tips:**\n"
            "- Ask clear, specific questions\n"
            "- Use the quiz to practice\n"
            "- Review answers for feedback"
        )


# ============ QUESTIONS PAGE ============
elif st.session_state.page == "questions":
    st.markdown("## ❓ Ask a Question")
    st.markdown("Ask the document anything and get an AI-powered answer with source citations.")

    query_input = st.text_input("Enter your question", value=st.session_state.query_text, key="query_input", placeholder="e.g., What is RAG?")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 Send Question", use_container_width=True, key="send_question"):
            if query_input.strip():
                with st.spinner("Finding answer..."):
                    answer, sources = generate_answer(
                        query_input,
                        st.session_state.retriever,
                        st.session_state.llm,
                        embeddings=st.session_state.embeddings,
                    )
                    st.session_state.answer = answer
                    st.session_state.sources = sources
                    st.session_state.query_text = query_input
            else:
                st.warning("Please enter a question")

    if st.session_state.answer:
        st.markdown("### Answer")
        st.success(st.session_state.answer)
        if st.session_state.sources:
            with st.expander("📌 View Sources", expanded=False):
                for idx, doc in enumerate(st.session_state.sources, 1):
                    st.markdown(f"**Source {idx}**")
                    st.write(doc.page_content[:300] + "...")


# ============ QUIZ PAGE ============
elif st.session_state.page == "quiz":
    st.markdown("## 📝 Interactive Quiz")
    st.markdown("Test your knowledge with AI-generated quiz questions.")

    if not st.session_state.quiz_data and not st.session_state.quiz_submitted:
        if st.button("🚀 Generate Quiz", use_container_width=True, key="create_quiz"):
            with st.spinner("Generating quiz..."):
                context = build_quiz_context(st.session_state.chunks)
                quiz_text = generate_quiz(context, st.session_state.llm)
                quiz_data = parse_quiz_output(quiz_text)
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_warning = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_feedback = []
                    st.rerun()
                else:
                    st.session_state.quiz_data = []
                    st.session_state.quiz_warning = "Could not parse the quiz. Please try again."

    if st.session_state.quiz_warning:
        st.warning(st.session_state.quiz_warning)

    if st.session_state.quiz_data and not st.session_state.quiz_submitted:
        st.markdown(f"**Questions: {len(st.session_state.quiz_data)}**")
        with st.form("quiz_form"):
            for idx, question in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{idx + 1}. {question['question']}**")
                st.radio(
                    "Select your answer:",
                    question["options"],
                    key=f"quiz_answer_{idx}",
                    label_visibility="collapsed"
                )
                st.markdown("---")

            submitted = st.form_submit_button("✅ Submit Quiz", use_container_width=True)

        if submitted:
            score = 0
            feedback = []
            for idx, question in enumerate(st.session_state.quiz_data):
                selected = st.session_state.get(f"quiz_answer_{idx}", "")
                answer_key = question["answer"].strip()
                correct_option = next(
                    (opt for opt in question["options"] if opt.startswith(answer_key)),
                    None,
                )
                correct_label = correct_option or question["answer"]
                is_correct = selected.startswith(answer_key) or selected == correct_label
                if is_correct:
                    score += 1
                feedback.append(
                    {
                        "question": question["question"],
                        "selected": selected,
                        "correct": correct_label,
                        "is_correct": is_correct,
                    }
                )

            st.session_state.quiz_submitted = True
            st.session_state.quiz_score = score
            st.session_state.quiz_feedback = feedback
            st.rerun()

    if st.session_state.quiz_submitted:
        st.markdown("## 📊 Quiz Results")
        total = len(st.session_state.quiz_data)
        percentage = (st.session_state.quiz_score / total * 100) if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", f"{st.session_state.quiz_score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage:.0f}%")
        with col3:
            if st.session_state.quiz_score == total:
                st.success("🎉 Perfect!")
            elif st.session_state.quiz_score >= total * 0.7:
                st.info("✓ Great job!")
            else:
                st.warning("Keep practicing!")

        st.markdown("---")
        st.markdown("### Answer Review")
        for idx, item in enumerate(st.session_state.quiz_feedback):
            st.markdown(f"**Q{idx + 1}. {item['question']}**")
            if item["is_correct"]:
                st.success(f"✓ Correct — {item['selected']}")
            else:
                st.error(f"✗ Incorrect — You chose: {item['selected']}")
                st.info(f"Correct answer: {item['correct']}")
            st.markdown("---")

        if st.button("🔄 Try Another Quiz", use_container_width=True, key="reset_quiz"):
            reset_quiz_state()
            st.rerun()
