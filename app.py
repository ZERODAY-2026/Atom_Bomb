import streamlit as st
import pymupdf
import analyze
import pdf
import safety
import storage
from datetime import datetime


# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="ClauseGuard | Legal Document Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optimization 1: Initialize database only once per session, not on every Streamlit rerun
if "db_initialized" not in st.session_state:
    storage.initialize_database()
    st.session_state.db_initialized = True


# ---------- CUSTOM STYLING ----------

st.markdown(
    """
    <style>
    /* ======================================================== */
    /* 1. WHITE / LIGHT BACKGROUNDS: Crisp Solid Black Font      */
    /* ======================================================== */

    /* Main application background: Pure clean white */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stMain"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* Headings on White Background: Pure Black Font */
    [data-testid="stMain"] h1:not(.hero h1),
    [data-testid="stMain"] h2:not(.hero h2),
    [data-testid="stMain"] h3:not(.hero h3),
    [data-testid="stMain"] h4:not(.hero h4),
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {
        color: #000000 !important;
        font-weight: 800 !important;
    }

    /* All text elements on White Background: Crisp Black Font */
    [data-testid="stMain"] p:not(.hero p),
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] small,
    [data-testid="stMain"] span:not(.hero *):not(button *):not([data-testid="stSidebar"] *) {
        color: #000000 !important;
        font-weight: 500;
    }

    /* Content cards & Bordered Containers on White Background */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #000000 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] * {
        color: #000000 !important;
    }

    /* Expanders on White Background */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #000000 !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #000000 !important;
    }

    /* Input fields and Textareas on White Background */
    .stTextInput input,
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1.5px solid #64748b !important;
        font-weight: 600 !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
    }

    /* File uploader container on White Background */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1.5px dashed #94a3b8 !important;
        border-radius: 12px !important;
        padding: 14px !important;
    }

    /* "Choose a document" Label (on white background) */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
    [data-testid="stFileUploader"] label * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Dark Blue Dropzone section: Light font for dark background */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a2538 !important;
        border: 1.5px solid #2d3f5e !important;
        border-radius: 10px !important;
    }

    /* All text inside the dark blue dropzone */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploadDropzone"],
    [data-testid="stFileUploadDropzone"] *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* Upload button inside dark dropzone: Dark button with crisp white font */
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploader"] button {
        background-color: #0b1526 !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] section button *,
    [data-testid="stFileUploadDropzone"] button *,
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Helper text (e.g. 200MB per file • PDF, TXT) on dark background */
    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] p {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    /* Upload icon inside dark dropzone */
    [data-testid="stFileUploader"] section svg,
    [data-testid="stFileUploadDropzone"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    /* Preformatted and Code blocks on White Background */
    .stApp pre,
    .stApp code,
    [data-testid="stText"] {
        background-color: #f8fafc !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Alerts (Success, Warning, Info) - High contrast black text */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stAlert"] * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Page Dividers */
    hr {
        border-color: #cbd5e1 !important;
    }


    /* ======================================================== */
    /* 2. DARK BACKGROUNDS: Crisp Light / White Font            */
    /* ======================================================== */

    /* Hero Banner: Dark Background with Light Font */
    .hero {
        background: linear-gradient(135deg, #091528 0%, #172c4c 100%) !important;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 18px rgba(9, 21, 40, 0.25);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 34px !important;
        font-weight: 800 !important;
        margin-bottom: 8px !important;
    }

    .hero p {
        color: #f1f5f9 !important;
        font-size: 16px !important;
        margin-bottom: 6px !important;
        font-weight: 400 !important;
    }

    .hero * {
        color: #ffffff !important;
    }

    /* Sidebar: Dark Background with Light Font */
    [data-testid="stSidebar"] {
        background-color: #0b1528 !important;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] caption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.18) !important;
    }

    /* Buttons: Dark Background with Light Font */
    .stButton > button,
    .stFormSubmitButton > button {
        color: #ffffff !important;
        background-color: #0f2b5c !important;
        border: 1.5px solid #1a3c6e !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.2rem !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: #ffffff !important;
        background-color: #173b75 !important;
        border-color: #2b579a !important;
    }

    .stButton > button *,
    .stFormSubmitButton > button * {
        color: #ffffff !important;
    }

    /* Speed badge */
    .perf-badge {
        display: inline-block;
        background: #e2fbe8;
        color: #000000 !important;
        border: 1px solid #147a32;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------- SESSION STATE ----------

defaults = {
    "document_text": None,
    "document_name": None,
    "pages": [],
    "document_injection_check": None,
    "summary": None,
    "current_answer": None,
    "current_question": None,
    "current_evidence_status": None,
    "current_evidence_message": None,
    "current_evidence_quote": None,
    "current_safety": None,
    "saved_answer_id": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- HELPER FUNCTIONS ----------

def extract_text_and_pages(uploaded_file):
    """
    Extract PDF or TXT text locally in memory.
    Returns both formatted document_text and structured pages list for passage retrieval.
    """
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise ValueError("This PDF is password-protected.")

            pages = []
            formatted_chunks = []

            for page_number, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append({"page": page_number, "text": page_text})
                    formatted_chunks.append(f"Page {page_number}:\n{page_text}")

            document_text = "\n\n".join(formatted_chunks)

    elif filename.endswith(".txt"):
        try:
            raw_text = file_bytes.decode("utf-8-sig").strip()
        except Exception:
            raw_text = file_bytes.decode("latin-1", errors="replace").strip()

        document_text = raw_text
        pages = [{"page": 1, "text": raw_text}] if raw_text else []

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not document_text.strip() or not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or contain no extractable text."
        )

    return document_text, pages


def extract_text(uploaded_file):
    """Backward-compatible wrapper returning document_text string."""
    doc_text, _ = extract_text_and_pages(uploaded_file)
    return doc_text


def get_evidence_quote(answer, evidence_check):
    """Return a quotation only when the verifier provides one."""
    if isinstance(evidence_check, dict):
        for key in ("quote", "evidence_quote", "supporting_quote"):
            value = evidence_check.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


# ---------- SIDEBAR ----------

with st.sidebar:
    st.markdown("## 🛡️ ClauseGuard")
    st.caption("Private legal document assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Document analysis", "Saved history"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Privacy & Performance")
    st.caption(
        "• Local Gemma 4 via Ollama (127.0.0.1)\n"
        "• In-memory passage retrieval\n"
        "• Model resident keep_alive enabled\n"
        "• Zero cloud telemetry"
    )

    st.divider()

    st.markdown("### Prototype status")
    st.caption(
        "Questions and answers you choose to save are stored "
        "in a local SQLite database."
    )


# ---------- HEADER ----------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ ClauseGuard</h1>
        <p>Private Legal Document Triage Agent</p>
        <p>Understand obligations, deadlines, and potential concerns
        with evidence-grounded AI assistance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Prototype only. Use dummy documents until privacy safeguards "
    "have been reviewed and tested."
)


# ============================================================
# DOCUMENT ANALYSIS PAGE
# ============================================================

if page == "Document analysis":

    st.markdown("## 1. Upload your agreement")
    st.caption("PDF or TXT · Extracted locally on this device")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt"],
        key="document_upload",
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: **{uploaded_file.name}**")

        btn_col1, btn_col2 = st.columns([1, 1])

        # Fast Instant Text Extraction (<0.1s)
        with btn_col1:
            extract_clicked = st.button(
                "⚡ Fast Extract Text (Instant Q&A)",
                type="primary",
                use_container_width=True,
            )

        # Full Document Extraction + Full AI Synthesis
        with btn_col2:
            full_analysis_clicked = st.button(
                "📄 Extract & Full 7-Section AI Analysis",
                use_container_width=True,
            )

        if extract_clicked or full_analysis_clicked:
            try:
                with st.spinner("Extracting text locally in memory..."):
                    document_text, pages = extract_text_and_pages(uploaded_file)

                # Store document and extracted pages
                st.session_state.document_text = document_text
                st.session_state.document_name = uploaded_file.name
                st.session_state.pages = pages
                st.session_state.summary = None
                st.session_state.current_answer = None
                st.session_state.current_question = None
                st.session_state.current_evidence_status = None
                st.session_state.current_evidence_message = None
                st.session_state.current_evidence_quote = None
                st.session_state.current_safety = None
                st.session_state.saved_answer_id = None

                # Optimization: Run injection check ONCE upon upload and cache it
                st.session_state.document_injection_check = safety.detect_prompt_injection(document_text)

                st.success(f"Extracted {len(pages)} page(s) successfully. Ready for instant Q&A below!")

                if full_analysis_clicked:
                    if not callable(getattr(analyze, "analyze_document", None)):
                        st.error("analyze.py does not expose analyze_document(document_text).")
                    else:
                        with st.spinner("Gemma is generating the comprehensive 7-section document overview..."):
                            summary = analyze.analyze_document(document_text)
                            st.session_state.summary = summary
                            st.success("Comprehensive document analysis completed.")
                            st.rerun()

            except Exception as error:
                st.error("Document extraction failed.")
                st.exception(error)

    # Document Overview Section
    if st.session_state.document_text:
        st.divider()

        st.markdown("## 2. Document overview")
        page_count = len(st.session_state.pages) if st.session_state.pages else 1
        st.caption(
            f"Loaded document: **{st.session_state.document_name}** ({page_count} pages) · "
            "⚡ In-memory passage indexing active"
        )

        with st.expander("View extracted document text"):
            st.text(st.session_state.document_text)

        # Optional on-demand comprehensive overview button if not generated yet
        if not st.session_state.summary:
            if st.button("Generate Full 7-Section AI Summary", type="secondary"):
                with st.spinner("Gemma is synthesizing your document summary..."):
                    try:
                        st.session_state.summary = analyze.analyze_document(st.session_state.document_text)
                        st.rerun()
                    except Exception as err:
                        st.error(f"Analysis failed: {err}")
        else:
            st.markdown("### AI document analysis")
            with st.container(border=True):
                st.markdown(st.session_state.summary)

    # Question Answering Section
    if st.session_state.document_text:
        st.divider()

        st.markdown("## 3. Ask Gemma")
        st.caption(
            "Ask about a clause, obligation, deadline, penalty, "
            "or termination condition."
        )

        with st.form("question_form"):
            question = st.text_input(
                "Your question",
                placeholder="Example: How much notice is required?",
            )

            ask_submitted = st.form_submit_button(
                "Ask Gemma",
                type="primary",
                use_container_width=True,
            )

        if ask_submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    q_clean = question.strip()

                    # Optimization: Use relevant passage retrieval if pages are available
                    pages = st.session_state.get("pages", [])
                    use_passages = False
                    relevant_passages = []

                    if pages and len(pages) > 0:
                        relevant_passages = pdf.find_relevant_passages(pages, q_clean, max_passages=5)
                        if relevant_passages:
                            use_passages = True

                    # Target context (targeted passages or full doc fallback)
                    if use_passages:
                        target_context = analyze.passages_to_document_text(relevant_passages)
                    else:
                        target_context = st.session_state.document_text

                    prompt = analyze.build_question_prompt(target_context, q_clean)

                    # Optimization: Live streaming for sub-second Time To First Token
                    st.markdown("### Generating answer...")
                    answer_container = st.empty()
                    collected_tokens = []

                    with st.spinner("Streaming response from local Gemma 4..."):
                        for token in analyze.ask_gemma_stream(prompt, num_predict=768):
                            collected_tokens.append(token)
                            answer_container.markdown("".join(collected_tokens) + "▌")

                    full_answer = "".join(collected_tokens)
                    answer_container.empty()

                    if not full_answer.strip():
                        # Fallback non-stream if stream returned empty
                        full_answer = analyze.ask_gemma(prompt, num_predict=768)

                    # Optimization: Verify evidence and safety with cached injection check
                    evidence = analyze.verify_answer_evidence(full_answer, st.session_state.document_text)
                    safety_result = safety.build_safety_notice(
                        q_clean,
                        st.session_state.document_text,
                        cached_injection_check=st.session_state.get("document_injection_check"),
                    )

                    status = (
                        "VERIFIED"
                        if evidence.get("status") == "VERIFIED"
                        else "UNVERIFIED"
                    )

                    st.session_state.current_safety = safety_result
                    st.session_state.current_answer = full_answer
                    st.session_state.current_question = q_clean
                    st.session_state.current_evidence_status = status
                    st.session_state.current_evidence_message = (
                        evidence.get(
                            "message",
                            "Quotation matched."
                            if status == "VERIFIED"
                            else "Evidence could not be verified.",
                        )
                    )
                    st.session_state.current_evidence_quote = (
                        get_evidence_quote(full_answer, evidence)
                    )
                    st.session_state.saved_answer_id = None

                except Exception as error:
                    st.error("Could not generate the answer.")
                    st.exception(error)

    # Display Current Answer
    if st.session_state.current_answer:
        st.divider()

        st.markdown("## 4. Your answer")
        safety_result = st.session_state.get("current_safety", {})

        for notice in safety_result.get("notices", []):
            st.warning(notice)

        with st.container(border=True):
            st.caption("QUESTION")
            st.markdown(st.session_state.current_question)

            st.markdown("### Explanation")
            st.markdown(st.session_state.current_answer)

            st.markdown("### Evidence check")

            if st.session_state.current_evidence_status == "VERIFIED":
                st.success(
                    "VERIFIED — "
                    + str(st.session_state.current_evidence_message)
                )
            else:
                st.warning(
                    "UNVERIFIED — "
                    + str(st.session_state.current_evidence_message)
                    + " Treat this answer cautiously."
                )

            if st.session_state.current_evidence_quote:
                st.markdown("**Supporting quotation identified**")
                st.info(st.session_state.current_evidence_quote)
            else:
                st.caption(
                    "No separate evidence quotation was provided "
                    "by the verification function."
                )

            st.caption(
                "A matching quotation does not establish that every "
                "claim in an answer is correct."
            )

        if st.session_state.saved_answer_id is not None:
            st.success(
                "This answer was saved. "
                f"Record ID: {st.session_state.saved_answer_id}"
            )

        elif st.button(
            "💾 Save answer to local history",
            type="primary",
        ):
            try:
                answer_id = storage.save_answer(
                    document_name=(
                        st.session_state.document_name
                        or "Unnamed document"
                    ),
                    question=st.session_state.current_question,
                    answer=st.session_state.current_answer,
                    evidence_status=(
                        st.session_state.current_evidence_status
                        or "UNVERIFIED"
                    ),
                    evidence_quote=(
                        st.session_state.current_evidence_quote
                    ),
                )

                st.session_state.saved_answer_id = answer_id
                st.success(
                    f"Answer saved locally. Record ID: {answer_id}"
                )

            except Exception as error:
                st.error("Could not save this answer.")
                st.exception(error)


# ============================================================
# SAVED HISTORY PAGE
# ============================================================

elif page == "Saved history":

    st.markdown("## 📚 Saved answer history")
    st.caption(
        "Review or delete answers saved in your local SQLite database."
    )

    try:
        history = storage.get_history(limit=100)

        if not history:
            st.info(
                "No saved answers yet. Ask a question and select "
                "'Save answer to local history' to save it."
            )

        else:
            st.caption("Showing up to 100 recent answers.")

            for item in history:
                record_id = item["id"]

                with st.container(border=True):
                    top_left, top_right = st.columns([4, 1])

                    with top_left:
                        st.markdown(f"**{item['document_name']}**")
                        st.caption(f"Record #{record_id}")

                    with top_right:
                        if item["evidence_status"] == "VERIFIED":
                            st.success("VERIFIED")
                        else:
                            st.warning("UNVERIFIED")

                    st.markdown("**Question**")
                    st.write(item["question"])

                    with st.expander("View saved answer"):
                        st.markdown(item["answer"])

                        if item.get("evidence_quote"):
                            st.markdown("**Supporting quotation**")
                            st.info(item["evidence_quote"])

                        created_at = item.get("created_at", "")

                        try:
                            display_time = datetime.fromisoformat(
                                created_at
                            ).astimezone().strftime(
                                "%d %b %Y, %I:%M %p"
                            )
                        except (ValueError, TypeError):
                            display_time = created_at

                        st.caption(f"Saved: {display_time}")

                    if st.button(
                        "🗑️ Delete this answer",
                        key=f"delete_answer_{record_id}",
                    ):
                        st.session_state[
                            f"confirm_delete_{record_id}"
                        ] = True

                    if st.session_state.get(
                        f"confirm_delete_{record_id}", False
                    ):
                        st.warning(
                            "Are you sure you want to delete this answer?"
                        )

                        confirm_col, cancel_col = st.columns(2)

                        with confirm_col:
                            if st.button(
                                "Yes, delete",
                                key=f"confirm_yes_{record_id}",
                                type="primary",
                            ):
                                storage.delete_answer(record_id)
                                st.session_state[
                                    f"confirm_delete_{record_id}"
                                ] = False
                                st.rerun()

                        with cancel_col:
                            if st.button(
                                "Cancel",
                                key=f"confirm_no_{record_id}",
                            ):
                                st.session_state[
                                    f"confirm_delete_{record_id}"
                                ] = False
                                st.rerun()

    except Exception as error:
        st.error("Could not load saved history.")
        st.exception(error)


# ---------- FOOTER ----------

st.divider()

st.caption(
    "ClauseGuard is a prototype, not a substitute for a qualified "
    "legal professional. Evidence matching is a basic check, not "
    "proof that an answer is complete or legally correct."
)
