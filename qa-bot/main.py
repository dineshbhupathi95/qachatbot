import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from loader import load_and_split_pdf
from vector_store import create_vector_store

st.title("📄 Document QA Bot (Program Manual Assistant)")

@st.cache_resource
def setup_bot():
    # Load PDF and create retriever
    docs = load_and_split_pdf("docs/UserManual.pdf")
    vectorstore = create_vector_store(docs)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Using 3 for most relevant docs

    # Load tokenizer and model from Hugging Face
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Create the pipeline with adjusted settings for concise responses
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

    # Build the RetrievalQA chain using the map_reduce method for more coherent answers
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Using map_reduce instead of stuff for more structured responses
        return_source_documents=True
    )

    return qa_chain

qa_chain = setup_bot()

query = st.text_input("Ask a question about the program manual:")
if query:
    with st.spinner("Thinking..."):
        prompt = f"""
        You are a knowledgeable assistant trained on the User Manual. Summarize the relevant sections of the manual to clearly and accurately answer the following question.

        Include key steps, and module names. Write as if explaining to a new user.

        Question: {query}
        """

        result = qa_chain(prompt)
        st.write("### 🤖 Answer:")
        st.write(result["result"])
        with st.expander("🗂️ Sources"):
            for doc in result['source_documents']:
                st.markdown(f"**Source Page:** {doc.metadata.get('page', '?')}")
                st.text(doc.page_content[:500])