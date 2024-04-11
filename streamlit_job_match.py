import pandas as pd
import os
import fitz
from pandas.io.parquet import json
import streamlit as st

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


def analyze_resume(job_desc, resume, scoring_prompt, criteria):
    options = [
        "Name",
        "Contact Number",
        "Years of Experience (Number)",
        "Highest Education",
        "Technical Skills",
        "Summary of all work experience including company's name and position held",
        "Achievements",
        "Research experience",
    ]
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
    scores = {}

    for criterion in criteria:
        score_question = (
            f"Job requirements: {{{job_desc}}}"
            + f"Resume summary: {{{df.to_string(index=False)}}}"
            + f"Given this scoring prompt: {{{scoring_prompt}}}"
            + f"Return only the score for this criterion: {{{criterion}}}"
        )
        score = ask_openAI(score_question)
        scores[criterion] = score

    for criterion, val in scores.items():
        df.loc[len(df)] = [criterion, val]
        print(val)

    final_score_question = (
        f"Given individual scores a resume receives for each criteria: {{{json.dumps(scores)}}}"
        + "What is the total score of this applicant? Give me only number and nothing else"
    )
    final_score = ask_openAI(final_score_question)
    df.loc[len(df)] = ["Match Score", final_score]

    return df


def ask_openAI(question):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a hiring manager",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0,
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
        response = chain.invoke(
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


def default_scoring_criteria():
    return """Scoring criteria based on the new priority:
    1. University education that is relevant to the job requirements: 4 points
    2. Work experience from a well-known company in the field. If the company is not well-known, score 0: 3 points
    3. Year of experience appropriate to the job description: 2 points
    4. Related skill sets specified in the job description such as programming languages, frameworks, tools: 1 point
    5. Any publication in terms of research or academic experiences: 0.5 points
Please return a matching score (0-10) for the candidate for this job, taking into account these criteria to facilitate comparison with other candidates.
Also, please include reasoning for each point that the applicant receives."""


def save_uploaded_file(directory, uploaded_file):
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def list_files(directory):
    files = []
    if os.path.isdir(directory):
        files = os.listdir(directory)
    return files


def delete_file(directory, filename):
    """Delete a specified file from a given directory."""
    file_path = os.path.join(directory, filename)
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file {filename}: {e}")
        return False


def analyze_multiple_resumes_with_criteria(job_desc, resume_texts, scoring_prompt):
    # Initialize an empty DataFrame for the compiled results
    compiled_results = pd.DataFrame()

    criteria_categorize_question = (
        f"Given a prompt used for scoring a resume on how compatible it is to a job description: {{{scoring_prompt}}} "
        + "Summarize and return only the criteria names in plain text and comma separated."
        + "Criteria are those that has number listing in front of them."
    )
    criteria = ask_openAI(criteria_categorize_question).split(", ")

    for _, resume_text in enumerate(tqdm(resume_texts, desc="Analyzing Resumes")):
        # Analyze each resume against the job description
        df = analyze_resume(job_desc, resume_text, scoring_prompt, criteria)

        # Extract summary and match score
        summary = df.loc[df["option"] == "Summary", "value"].values[0]
        # Assuming 'Name' is an option and it's unique for each resume
        candidate_name = df.loc[df["option"] == "Name", "value"].values[0]
        final_score = df.loc[df["option"] == "Match Score", "value"].values[0]
        result = {"Summary": summary, "Total": final_score}
        for criterion in criteria:
            criterion_score = df.loc[df["option"] == criterion, "value"].values[0]

            result[criterion] = criterion_score

        # Compile the individual resume results into the compiled DataFrame
        compiled_results[candidate_name] = pd.Series(result)

    return compiled_results


resume_directory = "resumes"
jd_directory = "job_descriptions"

st.title("Job Matching Service")

existing_resumes = list_files(resume_directory)
existing_jds = list_files(jd_directory)

uploaded_jd = st.file_uploader(
    "Upload a Job Description", type=["pdf"], key="jd_uploader"
)
if uploaded_jd:
    filename = save_uploaded_file(jd_directory, uploaded_jd)
    st.success(f"Uploaded {filename}")

# File uploader allows multiple files
uploaded_files = st.file_uploader("Upload new resumes", type=["pdf"])

# Save the uploaded files and refresh the list of available resumes
if uploaded_files:
    saved_file = save_uploaded_file(resume_directory, uploaded_files)
    st.success(f"Uploaded {saved_file} files.")

selected_resume_to_delete = st.selectbox(
    "Select a resume to delete", ["Select a file..."] + existing_resumes
)

if st.button("Delete existing resume"):
    if selected_resume_to_delete != "Select a file...":
        success = delete_file("resumes", selected_resume_to_delete)
        if success:
            st.success(f"Deleted {selected_resume_to_delete}")
        else:
            st.error("Failed to delete the resume.")
    else:
        st.error("Please select a valid resume to delete.")

selected_jd_to_delete = st.selectbox(
    "Select a job description to delete", ["Select a file..."] + existing_jds
)

if st.button("Delete existing job description"):
    if selected_jd_to_delete != "Select a file...":
        success = delete_file("job_descriptions", selected_jd_to_delete)
        if success:
            st.success(f"Deleted {selected_jd_to_delete}")
        else:
            st.error("Failed to delete the job description.")
    else:
        st.error("Please select a valid job description to delete.")

selected_resumes = st.multiselect("Select Resumes to Match", existing_resumes)
resume_texts = [
    extract_text_from_pdf(os.path.join(resume_directory, resume))
    for resume in selected_resumes
]

selected_jd = st.selectbox("Select a Job Description", existing_jds)
if selected_jd != "Please upload" and selected_jd is not None:
    jd_path = os.path.join(jd_directory, selected_jd)
    jd_text = extract_text_from_pdf(jd_path)

custom_scoring_prompt = st.text_area(
    "Specify Custom Scoring Criteria or Use Default",
    value=default_scoring_criteria(),
    height=300,
)

if st.button("Match and Score"):
    if selected_jd != "Please upload" and selected_resumes:
        # Assuming this function processes multiple resumes against a single JD and returns a DataFrame
        results_df = analyze_multiple_resumes_with_criteria(
            jd_text, resume_texts, custom_scoring_prompt
        )

        # Display results in a grid
        st.write("Matching Results")
        st.table(results_df)

        st.success("Analysis Complete.")
    else:
        st.error("Please select a job description and at least one resume.")
