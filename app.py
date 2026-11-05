import os 
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# load environment variables from .env file
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

#Initialize the chroma client with persistence
chroma_client= chromadb.PersistentClient(path = "chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(name= collection_name)

client = OpenAI(api_key= groq_key, base_url="https://api.groq.com/openai/v1")

# resp= client.chat.completions.create(
#     model= "openai/gpt-oss-20b",
#     messages= [
#         {"role": "system", "content": "you are a helpfull asistant"},
#         {"role": "user", "content": "What is human life expectancy in United State?"},
#     ],
# )

# print(resp.choices[0].message.content)

def load_documents_from_directory(directory_path):
    print("=========Loading Documents From Directory ==========")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding = "Utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

documents = load_documents_from_directory("news_articles")
print(f"Loading {len(documents)} Documents")

def split_text(text, chunk_size = 1000, chunk_overlap = 20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id" : f"{doc["id"]}_chunk{i+1}", "text" : chunk})

print(f"Total_chunks: {len(chunked_documents)}")

#donload the small chroma embedding function
for doc in chunked_documents:
    # print("=====Inserting chunks into db=====")
    collection.upsert(ids = [doc["id"]], documents = [doc["text"]])
# print(collection.get(limit = 1, include = ["embeddings"])["embeddings"])


# Function to query documents
def query_documents(questions, n_results = 2):
    results = collection.query(query_texts = [questions], n_results = n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist] 
    print("=====Returning relevent chunks=====")
    return relevant_chunks

#Function to generate a response from groq
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model = "openai/gpt-oss-20b",
        messages= [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content
    return answer
#Example query and response generation
question = "what is the biggest problem in tv"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)