import pandas as pd
import os
import fitz

from tqdm import tqdm
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


def analyze_resume(job_desc, resume, options):
    df = analyze_str(resume, options)
    df_string = df.apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    ).to_string(index=False)
    print("Analyzing with OpenAI...")
    summary_question = (
        f"Job requirements: {{{job_desc}}}"
        + f"Resume summary: {{{df_string}}}"
        + "Please return a summary of the candidate's suitability for this position (limited to 200 words).'"
    )
    summary = ask_openAI(summary_question)
    df.loc[len(df)] = ["Summary", summary]
    extra_info = """
    Scoring criteria based on the new priority:
    1. Top 10 universities in US/Canada: 4 points
    2. Work experience from an industry leader company: 3 points
    3. Work experience from a popular or well-known company: 2 points
    4. Related skill sets such as programming languages, frameworks, tools: 1 point
    5. Any publication in terms of research or academic experiences: 0.5 points
    Please return a matching score (0-10) for the candidate for this job, taking into account these criteria to facilitate comparison with other candidates.
    """
    score_question = (
        f"Job requirements: {{{job_desc}}}"
        + f"Resume summary: {{{df.to_string(index=False)}}}"
        + extra_info
    )
    score = ask_openAI(score_question)
    df.loc[len(df)] = ["Match Score", score]

    return df


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
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content


def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=600, chunk_overlap=100, length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    df_data = [{"option": option, "value": []} for option in options]
    print("Fetching information...")

    for i, option in tqdm(
        enumerate(options), desc="Fetching information", unit="option"
    ):
        question = f"What is this candidate's {option}? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"
        docs = knowledge_base.similarity_search(question)
        llm = langchain_openai(
            temperature=0.3,
        )
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain(
            {"input_documents": docs, "question": question}, return_only_outputs=True
        )
        df_data[i]["value"] = response["output_text"]

    df = pd.DataFrame(df_data)
    print("Resume elements retrieved successfully.")
    return df


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

    perfect_resume_text = """
        Jordan Smith

        Contact Information: [Email: jordan.smith@example.com | LinkedIn: linkedin.com/in/jordansmith]
        Objective: As a seasoned Data Scientist with over 8 years of experience in the tech industry, I am seeking to leverage my expertise in machine learning, data analysis, and project management to contribute to innovative projects at a leading technology firm.
        Education:

        Ph.D. in Computer Science, Stanford University, 2015
        Dissertation on "Advancements in Deep Learning Algorithms for Image Recognition"
        Published 5 papers in peer-reviewed journals on machine learning and AI.
        M.S. in Data Science, Massachusetts Institute of Technology (MIT), 2011
        B.S. in Computer Science, University of Toronto, 2009
        Graduated Summa Cum Laude
        Work Experience:

        Senior Data Scientist, Google, Mountain View, CA, 2018-Present
        Led a team of data scientists in developing machine learning models that improved search algorithm efficiency by 15%.
        Collaborated on cross-functional teams to integrate AI technologies into new product offerings.
        Data Scientist, Microsoft, Redmond, WA, 2015-2018
        Developed data processing pipelines that reduced data cleaning time by 30%.
        Implemented advanced predictive models for customer behavior that increased sales forecast accuracy by 20%.
        Data Analyst Intern, IBM, New York, NY, 2013-2014
        Supported data analysis and visualization projects that helped shape marketing strategies.
        Skills:

        Programming Languages: Proficient in Python, R, SQL, and Java.
        Tools & Technologies: Expertise in TensorFlow, PyTorch, Keras, Scikit-Learn, Tableau, and Power BI.
        Soft Skills: Strong analytical thinking, problem-solving capabilities, and excellent communication skills.
        Publications:

        Smith, J. (2016). "Enhancing Convolutional Neural Networks for Face Recognition", Journal of AI Research.
        Smith, J., & Doe, A. (2017). "Predictive Models for E-commerce: A Case Study", Data Science Quarterly.
        Certifications:

        Certified Data Scientist, Data Science Council of America (DASCA)
        Advanced Machine Learning, Coursera
        Professional Affiliations:

        Member, Association for Computing Machinery (ACM)
        Member, Institute of Electrical and Electronics Engineers (IEEE)
    """

    options = [
        "Name",
        "Contact Number",
        "Years of Work Experience (Number)",
        "Highest Education",
        "Undergraduate School Name",
        "Master's School Name",
        "Technical Skills",
        "Experience Level",
    ]

    # Call the analyze_resume function with default parameters for demonstration
    df = analyze_resume(job_description_text, perfect_resume_text, options)
    print(
        "Overall Match Score:", df.loc[df["option"] == "Match Score", "value"].values[0]
    )
    print("Detailed Display:")
    print(df)
