
import streamlit as st
import pymupdf
import analyze


# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="ClauseGuard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ClauseGuard")
st.subheader("Private Legal Document Triage Agent")

st.write(
    "Understand legal documents using a locally running Gemma model."
)

st.warning(
    "Prototype only. Use dummy documents until privacy safeguards "
    "have been reviewed and tested."
)


# ---------- PDF AND TXT EXTRACTION ----------

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

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

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


# ---------- SESSION STATE ----------

if "document_text" not in st.session_state:
    st.session_state.document_text = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "summary" not in st.session_state:
    st.session_state.summary = None


# ---------- UPLOAD DOCUMENT ----------

st.header("1. Upload a legal document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
    key="document_upload",
)

if uploaded_file is not None:

    if st.button("Extract and analyze document", type="primary"):

        try:
            with st.spinner("Extracting text locally..."):
                document_text = extract_text(uploaded_file)

            st.session_state.document_text = document_text
            st.session_state.document_name = uploaded_file.name
            st.session_state.summary = None

            if not callable(
                getattr(analyze, "analyze_document", None)
            ):
                st.error(
                    "Your analyze.py does not currently expose "
                    "analyze_document(document_text). Check its "
                    "function names before connecting the summary."
                )
            else:
                with st.spinner("Gemma is analyzing the document..."):
                    summary = analyze.analyze_document(document_text)

                st.session_state.summary = summary
                st.success("Analysis completed.")

        except Exception as error:
            st.error("Document analysis failed.")
            st.exception(error)


# ---------- SHOW EXTRACTED TEXT ----------

if st.session_state.document_text:

    st.caption(
        f"Loaded document: {st.session_state.document_name}"
    )

    with st.expander("View extracted document text"):
        st.text(st.session_state.document_text)


# ---------- SHOW SUMMARY ----------

if st.session_state.summary:

    st.header("2. Document analysis")
    st.markdown(st.session_state.summary)


# ---------- ASK QUESTIONS ----------

if st.session_state.document_text:

    st.header("3. Ask a question")

    question = st.text_input(
        "What would you like to understand?",
        placeholder="Example: How much notice is required?",
    )

    if st.button("Ask Gemma"):

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
                "Your analyze.py does not expose "
                "ask_about_document_with_verification(). "
                "We need to match the interface to its actual functions."
            )

        else:
            try:
                with st.spinner("Generating an answer locally..."):
                    result = (
                        analyze.ask_about_document_with_verification(
                            st.session_state.document_text,
                            question,
                        )
                    )

                st.subheader("Answer")
                st.markdown(result["answer"])

                st.subheader("Evidence verification")
                evidence = result["evidence_check"]

                if evidence.get("status") == "VERIFIED":
                    st.success(evidence.get("message", "Quotation matched."))
                else:
                    st.warning(
                        evidence.get(
                            "message",
                            "Evidence could not be verified.",
                        )
                        + " Treat this answer as unverified."
                    )

            except Exception as error:
                st.error("Could not generate the answer.")
                st.exception(error)


# ---------- FOOTER ----------

st.divider()

st.caption(
    "ClauseGuard is a prototype, not a substitute for a qualified "
    "legal professional. A quotation match does not prove that "
    "every claim in an answer is correct."
)
