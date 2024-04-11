import os
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI as langchain_openai
from langchain.chains.question_answering import load_qa_chain
from config import settings

os.environ["OPENAI_API_KEY"] = (
    settings.OPEN_AI_TOKEN if settings.OPEN_AI_TOKEN is not None else ""
)


# Function to extract text from the entire PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


resume_folder = "resumes"
pdf_files = [
    os.path.join(resume_folder, f)
    for f in os.listdir(resume_folder)
    if f.endswith(".pdf")
]

documents = [
    Document(page_content=extract_text_from_pdf(pdf_path)) for pdf_path in pdf_files
]

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_documents(documents, embeddings)
job_desc_text = extract_text_from_pdf(
    "job_descriptions/Job_Description_ Software_Developer.pdf"
)
question = f"{job_desc_text}"
docs = knowledge_base.similarity_search(question)
llm = langchain_openai(
    temperature=0.3,
)
chain = load_qa_chain(llm, chain_type="stuff")
response = chain.invoke(
    {"input_documents": docs, "question": question}, return_only_outputs=True
)
print(response["output_text"])
