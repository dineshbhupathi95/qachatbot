import streamlit as st
import tempfile
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from loader import load_and_split_pdf
from vector_store import create_vector_store

st.title("📄 Document QA Bot (Program Manual Assistant)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF to ask questions from:", type="pdf")

@st.cache_resource
def setup_bot(file=None):
    # Handle uploaded file or fallback to default
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        docs = load_and_split_pdf(tmp_file_path)
    else:
        docs = load_and_split_pdf("docs/UserManual.pdf")

    vectorstore = create_vector_store(docs)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=0.3,
        top_p=0.85,
        repetition_penalty=1.2,
        truncation=True,
        do_sample=False,
        device=-1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain

qa_chain = setup_bot(uploaded_file)

query = st.text_input("Ask a question about the document:")
if query:
    with st.spinner("Thinking..."):
        prompt = f"""
        You are a knowledgeable assistant trained on the uploaded document. Summarize the relevant sections clearly and accurately to answer the question below.

        Include key steps, module names, or concepts. Answer as if explaining to a new user.

        Question: {query}
        """

        result = qa_chain(prompt)
        st.write("### 🤖 Answer:")
        st.write(result["result"])

        with st.expander("🗂️ Sources", expanded=True):
            for doc in result['source_documents']:
                st.markdown(f"**Source Page:** {doc.metadata.get('page', '?')}")
                st.text(doc.page_content[:500])
