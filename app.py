
import streamlit as st  # pyright: ignore[reportMissingImports]
import pymupdf
import analyze
import storage
from datetime import datetime


# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="ClauseGuard | Legal Document Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

storage.initialize_database()


# ---------- CUSTOM STYLING ----------

st.markdown(
    """
    <style>
    /* Main page */
    .stApp {
        background-color: #f3f6fb;
        color: #172033;
    }

    [data-testid="stAppViewContainer"] .main {
        background-color: #f3f6fb;
    }

    [data-testid="stMain"] {
        color: #172033;
    }

    /* Default text color */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp .stMarkdown,
    .stApp [data-testid="stCaptionContainer"] {
        color: #273449;
    }

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #14213d !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #101b33;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f3f6ff !important;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #101b33, #294b7a);
        padding: 30px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 36px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #edf4ff !important;
        font-size: 16px;
        margin-bottom: 8px;
    }

    /* Content cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dce4ef;
        border-radius: 14px;
    }

    /* Input fields */
    .stTextInput input,
    .stTextArea textarea {
        color: #172033 !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #64748b !important;
        opacity: 1;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #dce4ef;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] section {
        color: #273449 !important;
    }

    /* Warning and info alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span {
        color: #273449 !important;
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        color: #ffffff !important;
        background-color: #244a80 !important;
        border: 1px solid #244a80 !important;
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        color: #ffffff !important;
        background-color: #18365f !important;
        border-color: #18365f !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #dce4ef;
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #172033 !important;
    }

    /* Saved answers and extracted text */
    .stApp pre,
    .stApp code {
        color: #172033;
    }

    /* Footer */
    .stApp footer {
        color: #475569;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- SESSION STATE ----------

defaults = {
    "document_text": None,
    "document_name": None,
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

def extract_text(uploaded_file):
    """Extract PDF or TXT text locally."""

    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        pdf = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )

        try:
            pages = []

            for page_idx in range(len(pdf)):
                page_number = page_idx + 1
                page = pdf[page_idx]
                page_text = str(page.get_text("text")).strip()

                if page_text:
                    pages.append(
                        f"Page {page_number}:\n{page_text}"
                    )

            document_text = "\n\n".join(pages)

        finally:
            pdf.close()

    elif filename.endswith(".txt"):
        document_text = file_bytes.decode("utf-8-sig")

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not document_text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or contain no extractable text."
        )

    return document_text


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

    st.markdown("### Privacy")
    st.caption(
        "Document extraction and model requests are intended "
        "to run locally. Verify your configuration before using "
        "real legal documents."
    )

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
        st.caption(f"Selected file: {uploaded_file.name}")

        if st.button(
            "Extract and analyze document",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Extracting text locally..."):
                    document_text = extract_text(uploaded_file)

                st.session_state.document_text = document_text
                st.session_state.document_name = uploaded_file.name
                st.session_state.summary = None
                st.session_state.current_answer = None
                st.session_state.current_question = None
                st.session_state.current_evidence_status = None
                st.session_state.current_evidence_message = None
                st.session_state.current_evidence_quote = None
                st.session_state.current_safety = None
                st.session_state.saved_answer_id = None

                if not callable(
                    getattr(analyze, "analyze_document", None)
                ):
                    st.error(
                        "analyze.py does not expose "
                        "analyze_document(document_text)."
                    )
                else:
                    with st.spinner(
                        "Gemma is analyzing the document locally..."
                    ):
                        summary = analyze.analyze_document(
                            document_text
                        )

                    st.session_state.summary = summary
                    st.success("Document analysis completed.")

            except Exception as error:
                st.error("Document analysis failed.")
                st.exception(error)

    if st.session_state.document_text:
        st.divider()

        st.markdown("## 2. Document overview")
        st.caption(
            f"Loaded document: {st.session_state.document_name}"
        )

        with st.expander("View extracted document text"):
            st.text(st.session_state.document_text)

    if st.session_state.summary:
        st.markdown("### AI document analysis")

        with st.container(border=True):
            st.markdown(st.session_state.summary)

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

            elif not callable(
                getattr(
                    analyze,
                    "ask_about_document_with_verification",
                    None,
                )
            ):
                st.error(
                    "analyze.py does not expose "
                    "ask_about_document_with_verification()."
                )

            else:
                try:
                    with st.spinner(
                        "Generating an answer using the local model..."
                    ):
                        result = (
                            analyze.ask_about_document_with_verification(
                                st.session_state.document_text,
                                question.strip(),
                            )
                        )

                    answer = result.get("answer", "")
                    evidence = result.get("evidence_check", {})

                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError(
                            "The model returned an empty answer."
                        )

                    if not isinstance(evidence, dict):
                        evidence = {}

                    status = (
                        "VERIFIED"
                        if evidence.get("status") == "VERIFIED"
                        else "UNVERIFIED"
                    )
                    safety_result = result.get("safety", {})
                    st.session_state.current_safety = safety_result
                    st.session_state.current_answer = answer
                    st.session_state.current_question = question.strip()
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
                        get_evidence_quote(answer, evidence)
                    )
                    st.session_state.saved_answer_id = None

                except Exception as error:
                    st.error("Could not generate the answer.")
                    st.exception(error)

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
