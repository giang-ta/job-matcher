# save documents as embeddings to DB

from typing import List
from markdown import Markdown
from io import StringIO
from pprint import pprint
from langchain.embeddings.openai import OpenAIEmbeddings
import hashlib

from file_joiner import traverse
from vectorstore import create_documents, Document


#### Markdown to Text
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# patching Markdown
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    return __md.convert(text)


#### END


def hash_str(string: str) -> str:
    return hashlib.md5(string.encode("utf-8")).hexdigest()


def read_md_file(file_path: str) -> str:
    with open(file_path, "r") as file_in:
        raw_content = file_in.read()
    content = unmark(raw_content)

    lines = []
    for line in content.split("\n"):
        lines.append(line)

    return "\n".join(lines)


def generate_documents(file_paths: list[str]) -> list[Document]:
    embeddings = OpenAIEmbeddings(
        openai_api_key="API_KEY",
    )
    documents = []

    for index, file_path in enumerate(file_paths):
        print(f'{index} / {len(file_paths)} - {file_path["file"]}')
        content = read_md_file(file_path["filepath"])
        document_id = hash_str(content)
        embedded_contents = embeddings.embed_documents([content])[0]

        documents.append(
            Document(
                source="support.justship.sg",
                content=content,
                documentId=document_id,
                embedding=embedded_contents,
            )
        )

    return documents


if __name__ == "__main__":
    path = "./data_source"
    file_paths = traverse(path)
    pprint(file_paths)

    documents = generate_documents(file_paths=file_paths)
    create_documents(documents)
