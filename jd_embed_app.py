import os
import fitz

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI as langchain_openai
from openai import OpenAI
from config import settings


openai_client = OpenAI(api_key=settings.OPEN_AI_TOKEN)
os.environ["OPENAI_API_KEY"] = (
    settings.OPEN_AI_TOKEN if settings.OPEN_AI_TOKEN is not None else ""
)


def analyze_resume(job_desc, resume):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=600, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    docs = knowledge_base.similarity_search(job_desc)
    llm = langchain_openai(
        temperature=0.3,
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    question = "Summarize this candidate resume with all important details about work experience, tools, programming languages, skills and achievements."
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    summary = response["output_text"]

    print(f"Extracted data from resume: {summary}")
    summary_question = (
        f"Given this job requirements: {job_desc}"
        + f"And resume summary: {summary}"
        + "Please return a summary of the candidate's suitability for this position. Do not make up skill that is not specified in the resume. (limited to 200 words).'"
    )
    response = chain(
        {"input_documents": docs, "question": summary_question},
        return_only_outputs=True,
    )
    open_ai_res = response["output_text"]
    print(open_ai_res)

    extra_info = """
    Scoring criteria based on the new priority:
    1. University education of at least bachelor degree: 4 points
    2. Work experience from an industry leader company: 3 points
    3. Work experience from a popular or well-known company: 2 points
    4. Related skill sets such as programming languages, frameworks, tools: 1 point
    5. Any publication in terms of research or academic experiences: 0.5 points
    Please return a matching score (0-10) for the candidate for this job, taking into account these criteria to facilitate comparison with other candidates.
    """

    extra_info_abstract = """
    Scoring criteria based on the new priority:
    1. Most year of experience working with technologies specified in the job description: 4 points
    2. Work experience from a popular well-known company: 3 points
    3. Demonstrate the problem-solving skill sets and the ability to think critically: 2 points
    4. Related skill sets such as programming languages, frameworks, tools: 1 point
    5. Any publication in terms of research or academic experiences: 0.5 points
    Please return a matching score (0-10) for the candidate for this job, taking into account these criteria to facilitate comparison with other candidates.
    """
    score_question = (
        f"Job requirements: {{{job_desc}}}"
        + f"Resume summary: {{{open_ai_res}}}"
        + extra_info_abstract
    )
    score = ask_openAI(score_question)

    return score


def ask_openAI(question):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# Example usage
if __name__ == "__main__":
    # Path to the job description and resume PDF files
    job_description_pdf_path = "jd.pdf"
    resume_pdf_path = "resume.pdf"

    # Extract text from PDFs
    job_description_text = extract_text_from_pdf(job_description_pdf_path)
    resume_text = extract_text_from_pdf(resume_pdf_path)

    # Call the analyze_resume function with default parameters for demonstration
    score = analyze_resume(job_description_text, resume_text)
    print("Overall Match Score: ", score)
