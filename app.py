import streamlit as st
from src.loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, load_vector_store
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_pipeline import generate_answer
from src.quiz_generator import generate_quiz, parse_quiz_output
from agent_etudiant_tp import run_agent as run_study_agent

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
    st.session_state.store_type = "faiss"  # Vector store type
    st.session_state.agent_response = None
    st.session_state.agent_query = ""


def reset_quiz_state():
    st.session_state.quiz_text = None
    st.session_state.quiz_data = []
    st.session_state.quiz_warning = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = []


def build_quiz_context(chunks, max_total_chars=400, max_chunks=1, max_chars_per_chunk=400):
    """Build minimal context for quiz to avoid 413 errors."""
    if chunks:
        text = chunks[0].page_content.strip()[:max_chars_per_chunk]
        return text

    # If no chunks available, try to pull from a loaded retriever
    retriever = st.session_state.get("retriever")
    if retriever:
        try:
            docs = retriever.get_relevant_documents("summary")
            if docs:
                return docs[0].page_content.strip()[:max_chars_per_chunk]
        except Exception:
            pass

    return ""


# SIDEBAR: PDF Upload
with st.sidebar:
    st.subheader("📄 Document Upload")
    
    # Vector store selector
    st.session_state.store_type = st.radio(
        "Vector Store:",
        ["faiss", "chroma"],
        horizontal=True,
        help="FAISS: Fast, memory-based. Chroma: Persistent, lightweight."
    )
    # Option to load an already-built store from disk
    if st.button("📥 Load existing store", key="load_store"):
        with st.spinner("Loading store..."):
            embeddings = get_embeddings()
            try:
                if st.session_state.store_type == "chroma":
                    db = load_vector_store("embeddings/chroma_db", embeddings, store_type="chroma")
                else:
                    db = load_vector_store("embeddings/faiss_index", embeddings, store_type="faiss")

                retriever = get_retriever(db, search_k=20)
                llm = get_llm()

                st.session_state.embeddings = embeddings
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                # Mark as 'loaded' so navigation and features are available
                st.session_state.pdf_loaded = True
                st.session_state.uploaded_file_name = f"Loaded {st.session_state.store_type.upper()} store"
                st.success(f"Loaded existing {st.session_state.store_type.upper()} store")
            except Exception as e:
                st.error(f"Could not load store: {e}")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    append_to_store = st.checkbox("Append to existing store", value=False, help="If checked, new PDF contents will be added to the selected vector store instead of replacing it.")

    if uploaded_file:
        if not st.session_state.pdf_loaded or st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with open("data/temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                docs = load_pdf("data/temp.pdf")
                chunks = split_documents(docs)

                embeddings = get_embeddings()
                db = create_vector_store(
                    chunks,
                    embeddings,
                    store_type=st.session_state.store_type,
                    append=append_to_store,
                )
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
            st.success(f"✓ Loaded: {uploaded_file.name} ({st.session_state.store_type.upper()})")

    if st.session_state.pdf_loaded:
        st.info(f"**Current PDF:** {st.session_state.uploaded_file_name}\n**Store:** {st.session_state.store_type.upper()}")


# PAGE NAVIGATION
if st.session_state.pdf_loaded:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
    with col2:
        if st.button("❓ Ask Question", use_container_width=True):
            st.session_state.page = "questions"
    with col3:
        if st.button("📝 Take Quiz", use_container_width=True):
            st.session_state.page = "quiz"
    with col4:
        if st.button("🤖 Agent", use_container_width=True):
            st.session_state.page = "agent"
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
            if not query_input.strip():
                st.warning("Please enter a question")
            else:
                # Ensure retriever & llm available; try to auto-load store if missing
                if not st.session_state.get("retriever"):
                    with st.spinner("Loading existing store..."):
                        try:
                            embeddings = get_embeddings()
                            if st.session_state.store_type == "chroma":
                                db = load_vector_store("embeddings/chroma_db", embeddings, store_type="chroma")
                            else:
                                db = load_vector_store("embeddings/faiss_index", embeddings, store_type="faiss")
                            st.session_state.retriever = get_retriever(db, search_k=20)
                            st.session_state.embeddings = embeddings
                            st.session_state.llm = get_llm()
                        except Exception as e:
                            st.error(f"No vector store available: {e}")
                if not st.session_state.get("retriever"):
                    st.warning("No vector store loaded. Upload a PDF or Load existing store first.")
                else:
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

    topic_input = st.text_input("Enter a topic for the quiz (optional)", key="quiz_topic_input", placeholder="e.g., \"the history of artificial intelligence\"")

    if not st.session_state.quiz_data and not st.session_state.quiz_submitted:
        if st.button("🚀 Generate Quiz", use_container_width=True, key="create_quiz"):
            with st.spinner("Generating quiz..."):
                context = build_quiz_context(st.session_state.chunks)
                if not st.session_state.get("llm"):
                    st.session_state.llm = get_llm()
                quiz_text = generate_quiz(context, st.session_state.llm, topic=topic_input)
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


# ============ AGENT PAGE ============
elif st.session_state.page == "agent":
    st.markdown("## 🤖 Study Agent")
    st.markdown("Ask your study agent for help. It can:\n- **Calculate averages** from test scores\n- **Search your notes** for specific topics\n- **Generate revision plans** for any subject")

    agent_input = st.text_input(
        "Ask your study agent:",
        value=st.session_state.agent_query,
        key="agent_input",
        placeholder="e.g., 'Calculate the average of [12, 15, 9]' or 'Search for RAG in notes'"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💭 Send", use_container_width=True, key="send_agent"):
            if not agent_input.strip():
                st.warning("Please enter a question for the agent.")
            else:
                with st.spinner("Agent is thinking..."):
                    try:
                        response = run_study_agent(agent_input)
                        st.session_state.agent_response = response
                        st.session_state.agent_query = agent_input
                    except Exception as e:
                        st.error(f"Agent error: {e}")

    if st.session_state.agent_response:
        st.markdown("### Agent Response")
        st.success(st.session_state.agent_response)

    st.markdown("---")
    st.markdown("### Example Queries")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Calculate [12, 15, 9]", use_container_width=True, key="example_calc"):
            st.session_state.agent_query = "Calcule la moyenne de [12, 15, 9]."
            with st.spinner("Agent is thinking..."):
                try:
                    response = run_study_agent(st.session_state.agent_query)
                    st.session_state.agent_response = response
                except Exception as e:
                    st.error(f"Agent error: {e}")
            st.rerun()
    with col2:
        if st.button("🔍 Search notes", use_container_width=True, key="example_search"):
            st.session_state.agent_query = "Cherche le mot RAG dans mes notes."
            with st.spinner("Agent is thinking..."):
                try:
                    response = run_study_agent(st.session_state.agent_query)
                    st.session_state.agent_response = response
                except Exception as e:
                    st.error(f"Agent error: {e}")
            st.rerun()
    with col3:
        if st.button("📚 Revision plan", use_container_width=True, key="example_revision"):
            st.session_state.agent_query = "Prépare-moi un plan de révision sur les agents IA."
            with st.spinner("Agent is thinking..."):
                try:
                    response = run_study_agent(st.session_state.agent_query)
                    st.session_state.agent_response = response
                except Exception as e:
                    st.error(f"Agent error: {e}")
            st.rerun()
