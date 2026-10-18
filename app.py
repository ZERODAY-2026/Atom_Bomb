import json
import sys
from datetime import datetime
from pathlib import Path

import pymupdf
import streamlit as st  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import analyze
import eval_cuad  # pyright: ignore[reportMissingImports]
import storage

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
    "cuad_contracts": None,
    "cuad_questions": None,
    "cuad_benchmark_results": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- HELPER FUNCTIONS ----------

def extract_text(uploaded_file):
    """
    Extract text locally from PDF, TXT, or JSON documents.
    Returns either a string for single documents or a dict for CUAD datasets.
    """
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
                    pages.append(f"Page {page_number}:\n{page_text}")

            document_text = "\n\n".join(pages)
        finally:
            pdf.close()

        if not document_text.strip():
            raise ValueError(
                "No readable text found. The PDF may be scanned "
                "or contain no extractable text."
            )
        return document_text

    elif filename.endswith(".txt") or filename.endswith(".md"):
        document_text = file_bytes.decode("utf-8-sig")
        if not document_text.strip():
            raise ValueError("The uploaded text document is empty.")
        return document_text

    elif filename.endswith(".json"):
        try:
            raw_data = json.loads(file_bytes.decode("utf-8-sig"))
        except Exception as err:
            raise ValueError(f"Could not parse JSON file: {err}")

        # Check if it's a CUAD SQuAD-format dataset
        if isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], list):
            contracts = []
            for idx, item in enumerate(raw_data["data"]):
                title = item.get("title") or f"Contract {idx + 1}"
                paras = item.get("paragraphs", [])
                full_text = "\n\n".join(
                    p.get("context", "") for p in paras if p.get("context")
                ).strip()
                all_qas = [qa for p in paras for qa in p.get("qas", [])]
                if full_text:
                    contracts.append({
                        "title": title,
                        "text": full_text,
                        "qas": all_qas,
                    })

            if not contracts:
                raise ValueError("No contracts with text found in this CUAD JSON dataset.")

            return {
                "type": "cuad",
                "contracts": contracts,
                "raw_data": raw_data,
            }

        # Single JSON contract structure
        if isinstance(raw_data, dict):
            for key in ("text", "content", "contract", "context", "document", "agreement"):
                if isinstance(raw_data.get(key), str) and raw_data[key].strip():
                    return raw_data[key].strip()

            formatted = [f"{k.upper()}:\n{v}" for k, v in raw_data.items() if str(v).strip()]
            if formatted:
                return "\n\n".join(formatted)

        elif isinstance(raw_data, list):
            content = "\n\n".join(str(x) for x in raw_data if str(x).strip())
            if content.strip():
                return content

        raise ValueError("The JSON file does not contain readable legal document text.")

    else:
        raise ValueError("Only PDF, TXT, and JSON files are supported.")


def get_evidence_quote(answer, evidence_check):
    """Return a quotation only when the verifier provides one."""
    if isinstance(evidence_check, dict):
        for key in ("quote", "evidence_quote", "supporting_quote"):
            value = evidence_check.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def get_builtin_demo_dataset():
    """Built-in CUAD benchmark contracts with expert annotations for instant testing."""
    return {
        "data": [
            {
                "title": "Distribution Agreement (Commercial)",
                "paragraphs": [{
                    "context": (
                        "DISTRIBUTION AGREEMENT\n"
                        "This Agreement is governed by the laws of the State of California.\n"
                        "The Distributor shall pay all invoices within 30 days of receipt.\n"
                        "Either party may terminate this Agreement with 60 days written notice.\n"
                        "During the term and for 12 months thereafter, Distributor shall not distribute competing products.\n"
                    ),
                    "qas": [
                        {
                            "id": "T__Governing Law",
                            "question": 'Highlight the parts related to "Governing Law" that should be reviewed by a lawyer. Details: Which state or country\'s law governs the contract?',
                            "answers": [{"text": "governed by the laws of the State of California", "answer_start": 0}],
                        },
                        {
                            "id": "T__Notice Period To Terminate Renewal",
                            "question": 'Highlight the parts related to "Termination" that should be reviewed by a lawyer. Details: notice to terminate',
                            "answers": [{"text": "terminate this Agreement with 60 days written notice", "answer_start": 0}],
                        },
                        {
                            "id": "T__Non-Compete",
                            "question": 'Highlight the parts related to "Non-Compete" that should be reviewed by a lawyer. Details: restrictions on competing products',
                            "answers": [{"text": "Distributor shall not distribute competing products", "answer_start": 0}],
                        },
                    ],
                }],
            },
            {
                "title": "Non-Disclosure and Confidentiality Agreement",
                "paragraphs": [{
                    "context": (
                        "MUTUAL NON-DISCLOSURE AGREEMENT\n"
                        "The Receiving Party agrees to protect Confidential Information for a period of five (5) years.\n"
                        "Neither party may assign this Agreement without the prior written consent of the other party.\n"
                        "Each party shall defend and indemnify the other against third party claims resulting from breach.\n"
                    ),
                    "qas": [
                        {
                            "id": "NDA__Anti-Assignment",
                            "question": 'Highlight the parts related to "Anti-Assignment" that should be reviewed by a lawyer. Details: restrictions on assigning rights or obligations',
                            "answers": [{"text": "Neither party may assign this Agreement without the prior written consent", "answer_start": 0}],
                        },
                        {
                            "id": "NDA__Indemnification",
                            "question": 'Highlight the parts related to "Indemnification" that should be reviewed by a lawyer. Details: indemnifying or holding harmless',
                            "answers": [{"text": "defend and indemnify the other against third party claims", "answer_start": 0}],
                        },
                        {
                            "id": "NDA__Solar Panel Warranty",
                            "question": 'Highlight the parts related to "Solar Panel Warranty" that should be reviewed by a lawyer.',
                            "answers": [],
                        },
                    ],
                }],
            },
        ]
    }


# ---------- SIDEBAR ----------

with st.sidebar:
    st.markdown("## 🛡️ ClauseGuard")
    st.caption("Private legal document assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Document analysis", "Saved history", "CUAD Benchmark"],
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
    st.caption("PDF, TXT, or JSON (including CUAD datasets) · Extracted locally on this device")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt", "json"],
        key="document_upload",
    )

    if uploaded_file is not None:
        st.caption(f"Selected file: {uploaded_file.name}")

        try:
            extracted = extract_text(uploaded_file)

            # If the user uploaded a CUAD dataset containing multiple contracts
            if isinstance(extracted, dict) and extracted.get("type") == "cuad":
                contracts = extracted["contracts"]
                st.info(f"📂 CUAD Dataset detected: Found {len(contracts)} contracts with expert lawyer annotations.")

                contract_labels = [f"{i + 1}. {c['title']}" for i, c in enumerate(contracts)]
                selected_idx = st.selectbox(
                    "Select a contract from this dataset to analyze:",
                    range(len(contracts)),
                    format_func=lambda i: contract_labels[i],
                    key="cuad_contract_picker",
                )

                chosen_contract = contracts[selected_idx]
                qa_count = len(chosen_contract.get("qas", []))
                st.caption(f"Contract text: ~{len(chosen_contract['text'])} characters · {qa_count} labelled questions available")

                if st.button(
                    "Extract and analyze selected contract",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        st.session_state.document_text = chosen_contract["text"]
                        st.session_state.document_name = f"{uploaded_file.name} — {chosen_contract['title']}"
                        st.session_state.cuad_questions = chosen_contract.get("qas", [])
                        st.session_state.summary = None
                        st.session_state.current_answer = None
                        st.session_state.current_question = None
                        st.session_state.current_evidence_status = None
                        st.session_state.current_evidence_message = None
                        st.session_state.current_evidence_quote = None
                        st.session_state.current_safety = None
                        st.session_state.saved_answer_id = None

                        with st.spinner("Gemma is analyzing the contract locally..."):
                            summary = analyze.analyze_document(chosen_contract["text"])

                        st.session_state.summary = summary
                        st.success(f"Analysis completed for: {chosen_contract['title']}")
                        st.rerun()
                    except Exception as error:
                        st.error("Document analysis failed.")
                        st.exception(error)

            else:
                # Single document (PDF, TXT, or single JSON)
                if st.button(
                    "Extract and analyze document",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        st.session_state.document_text = str(extracted)
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.cuad_questions = None
                        st.session_state.summary = None
                        st.session_state.current_answer = None
                        st.session_state.current_question = None
                        st.session_state.current_evidence_status = None
                        st.session_state.current_evidence_message = None
                        st.session_state.current_evidence_quote = None
                        st.session_state.current_safety = None
                        st.session_state.saved_answer_id = None

                        if not callable(getattr(analyze, "analyze_document", None)):
                            st.error("analyze.py does not expose analyze_document(document_text).")
                        else:
                            with st.spinner("Gemma is analyzing the document locally..."):
                                summary = analyze.analyze_document(str(extracted))

                            st.session_state.summary = summary
                            st.success("Document analysis completed.")

                    except Exception as error:
                        st.error("Document analysis failed.")
                        st.exception(error)

        except Exception as error:
            st.error("Could not process this file.")
            st.exception(error)

    if st.session_state.document_text:
        st.divider()

        st.markdown("## 2. Document overview")
        st.caption(f"Loaded document: {st.session_state.document_name}")

        with st.expander("View extracted document text"):
            st.text(st.session_state.document_text)

    if st.session_state.summary:
        st.markdown("### AI document analysis")
        with st.container(border=True):
            st.markdown(st.session_state.summary)

    if st.session_state.document_text:
        st.divider()

        st.markdown("## 3. Ask Gemma")
        st.caption("Ask about a clause, obligation, deadline, penalty, or termination condition.")

        # If this contract has labelled benchmark questions from CUAD, provide quick pick
        preset_question = ""
        if st.session_state.get("cuad_questions"):
            qas = [qa for qa in st.session_state["cuad_questions"] if qa.get("question")]
            if qas:
                q_options = ["(Custom question)"] + [qa.get("question", "") for qa in qas[:15]]
                picked = st.selectbox("💡 Quick pick: Labelled CUAD question for this contract:", q_options)
                if picked != "(Custom question)":
                    preset_question = picked

        with st.form("question_form"):
            question = st.text_input(
                "Your question",
                value=preset_question,
                placeholder="Example: How much notice is required to terminate?",
            )

            ask_submitted = st.form_submit_button(
                "Ask Gemma",
                type="primary",
                use_container_width=True,
            )

        if ask_submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            elif not callable(getattr(analyze, "ask_about_document_with_verification", None)):
                st.error("analyze.py does not expose ask_about_document_with_verification().")
            else:
                try:
                    with st.spinner("Generating an answer using the local model..."):
                        result = analyze.ask_about_document_with_verification(
                            st.session_state.document_text,
                            question.strip(),
                        )

                    answer = result.get("answer", "")
                    evidence = result.get("evidence_check", {})

                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("The model returned an empty answer.")

                    if not isinstance(evidence, dict):
                        evidence = {}

                    status = "VERIFIED" if evidence.get("status") == "VERIFIED" else "UNVERIFIED"
                    safety_result = result.get("safety", {})

                    st.session_state.current_safety = safety_result
                    st.session_state.current_answer = answer
                    st.session_state.current_question = question.strip()
                    st.session_state.current_evidence_status = status
                    st.session_state.current_evidence_message = (
                        evidence.get(
                            "message",
                            "Quotation matched." if status == "VERIFIED" else "Evidence could not be verified.",
                        )
                    )
                    st.session_state.current_evidence_quote = get_evidence_quote(answer, evidence)
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
                st.success("VERIFIED — " + str(st.session_state.current_evidence_message))
            else:
                st.warning("UNVERIFIED — " + str(st.session_state.current_evidence_message) + " Treat this answer cautiously.")

            if st.session_state.current_evidence_quote:
                st.markdown("**Supporting quotation identified**")
                st.info(st.session_state.current_evidence_quote)
            else:
                st.caption("No separate evidence quotation was provided by the verification function.")

            st.caption("A matching quotation does not establish that every claim in an answer is correct.")

        if st.session_state.saved_answer_id is not None:
            st.success(f"This answer was saved. Record ID: {st.session_state.saved_answer_id}")
        elif st.button("💾 Save answer to local history", type="primary"):
            try:
                answer_id = storage.save_answer(
                    document_name=(st.session_state.document_name or "Unnamed document"),
                    question=st.session_state.current_question,
                    answer=st.session_state.current_answer,
                    evidence_status=(st.session_state.current_evidence_status or "UNVERIFIED"),
                    evidence_quote=(st.session_state.current_evidence_quote),
                )
                st.session_state.saved_answer_id = answer_id
                st.success(f"Answer saved locally. Record ID: {answer_id}")
            except Exception as error:
                st.error("Could not save this answer.")
                st.exception(error)


# ============================================================
# SAVED HISTORY PAGE
# ============================================================

elif page == "Saved history":

    st.markdown("## 📚 Saved answer history")
    st.caption("Review or delete answers saved in your local SQLite database.")

    try:
        history = storage.get_history(limit=100)

        if not history:
            st.info("No saved answers yet. Ask a question and select 'Save answer to local history' to save it.")
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
                            display_time = datetime.fromisoformat(created_at).astimezone().strftime("%d %b %Y, %I:%M %p")
                        except (ValueError, TypeError):
                            display_time = created_at

                        st.caption(f"Saved: {display_time}")

                    if st.button("🗑️ Delete this answer", key=f"delete_answer_{record_id}"):
                        st.session_state[f"confirm_delete_{record_id}"] = True

                    if st.session_state.get(f"confirm_delete_{record_id}", False):
                        st.warning("Are you sure you want to delete this answer?")
                        confirm_col, cancel_col = st.columns(2)

                        with confirm_col:
                            if st.button("Yes, delete", key=f"confirm_yes_{record_id}", type="primary"):
                                storage.delete_answer(record_id)
                                st.session_state[f"confirm_delete_{record_id}"] = False
                                st.rerun()

                        with cancel_col:
                            if st.button("Cancel", key=f"confirm_no_{record_id}"):
                                st.session_state[f"confirm_delete_{record_id}"] = False
                                st.rerun()

    except Exception as error:
        st.error("Could not load saved history.")
        st.exception(error)


# ============================================================
# CUAD BENCHMARK PAGE
# ============================================================

elif page == "CUAD Benchmark":

    st.markdown("## 📊 CUAD Benchmark & Retrieval Evaluation")
    st.caption(
        "Evaluate ClauseGuard's BM25 passage retrieval on expert-annotated commercial "
        "contracts from The Atticus Project (CUAD, NeurIPS 2021)."
    )

    with st.container(border=True):
        st.markdown("### 1. Dataset Configuration")

        dataset_choice = st.radio(
            "Select evaluation dataset:",
            ["Use built-in benchmark contracts (Instant Demo)", "Upload custom CUAD JSON dataset"],
            horizontal=True,
        )

        dataset_to_eval = None

        if dataset_choice == "Upload custom CUAD JSON dataset":
            cuad_file = st.file_uploader(
                "Upload CUAD JSON file (e.g., CUAD_v1.json or test.json)",
                type=["json"],
                key="cuad_benchmark_upload",
            )
            if cuad_file is not None:
                try:
                    dataset_to_eval = json.loads(cuad_file.getvalue().decode("utf-8-sig"))
                    num_contracts = len(dataset_to_eval.get("data", []))
                    st.success(f"Loaded dataset: `{cuad_file.name}` with {num_contracts} contracts.")
                except Exception as exc:
                    st.error(f"Invalid JSON file: {exc}")
        else:
            dataset_to_eval = get_builtin_demo_dataset()
            st.info("Using built-in multi-clause commercial contracts with expert labels.")

        col1, col2 = st.columns(2)
        with col1:
            top_k_val = st.slider("Top-k passages retrieved per question", min_value=1, max_value=10, value=5)
        with col2:
            max_c_val = st.slider("Max contracts to evaluate", min_value=1, max_value=50, value=10)

        run_eval_clicked = st.button("🚀 Run CUAD Retrieval Benchmark", type="primary", use_container_width=True)

        if run_eval_clicked:
            if not dataset_to_eval or not dataset_to_eval.get("data"):
                st.warning("Please select or upload a valid CUAD dataset first.")
            else:
                progress_bar = st.progress(0, text="Evaluating contracts...")

                def on_progress(current, total):
                    pct = current / total if total > 0 else 1.0
                    progress_bar.progress(pct, text=f"Evaluating contract {current} of {total}...")

                with st.spinner("Running retrieval evaluation across legal clauses..."):
                    eval_result = eval_cuad.evaluate(
                        dataset_to_eval,
                        top_k=top_k_val,
                        max_contracts=max_c_val,
                        progress_callback=on_progress,
                    )

                progress_bar.empty()
                st.session_state.cuad_benchmark_results = eval_result
                st.success("CUAD Benchmark evaluation complete!")

    if st.session_state.cuad_benchmark_results:
        res = st.session_state.cuad_benchmark_results
        st.divider()

        st.markdown("### 2. Retrieval Accuracy Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contracts Tested", res["contracts"])
        m2.metric("Labelled Questions", res["labelled_questions"])
        m3.metric("Passage Hits", res["hits"])
        hit_rate_pct = (res.get("hit_rate") or 0.0) * 100
        m4.metric("Overall Hit Rate", f"{hit_rate_pct:.1f}%")

        if res.get("unanswerable_questions"):
            unans_rate = (
                res["unanswerable_with_results"] / res["unanswerable_questions"]
                if res["unanswerable_questions"]
                else 0
            )
            st.caption(
                f"Off-topic rejection rate: {res['unanswerable_with_results']} of "
                f"{res['unanswerable_questions']} unanswerable questions returned passages ({unans_rate * 100:.1f}%)."
            )

        st.markdown("#### Performance by Clause Type")

        table_rows = []
        for clause_name, stats in res.get("by_clause_type", {}).items():
            rate = stats.get("hit_rate", 0.0) * 100
            table_rows.append({
                "Clause Type": clause_name,
                "Questions Asked": stats["asked"],
                "Hits": stats["hits"],
                "Hit Rate (%)": f"{rate:.1f}%",
            })

        if table_rows:
            st.table(table_rows)

        st.download_button(
            "📥 Download Benchmark Report (JSON)",
            data=json.dumps(res, indent=2),
            file_name="cuad_retrieval_eval.json",
            mime="application/json",
        )


# ---------- FOOTER ----------

st.divider()

st.caption(
    "ClauseGuard is a prototype, not a substitute for a qualified "
    "legal professional. Evidence matching is a basic check, not "
    "proof that an answer is complete or legally correct."
)
