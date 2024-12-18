import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    # Set a user-agent to simulate a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Send GET request to the website with headers
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all textual content from paragraphs
        text = " ".join([p.get_text() for p in soup.find_all('p')])
        return text
    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None

# Example usage
url = "https://www.uchicago.edu/"
website_text = scrape_website(url)

if website_text:
    print(website_text[:500])  # Print first 500 characters of the scraped text
else:
    print("Failed to retrieve website content.")
def chunk_text(text, chunk_size=500):
    # Split text into chunks of a specific size
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Example usage
chunks = chunk_text(website_text)
print(chunks[:2])  # Print first 2 chunks
from sentence_transformers import SentenceTransformer

# Initialize pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    # Convert each chunk to embeddings
    embeddings = model.encode(chunks)
    return embeddings

# Example usage
chunk_embeddings = get_embeddings(chunks)
print(chunk_embeddings[:2])  # Print the first two embeddings
import faiss
import numpy as np

def create_faiss_index(embeddings):
    # Convert embeddings to numpy array (FAISS requires numpy arrays)
    embeddings_np = np.array(embeddings).astype(np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance metric
    index.add(embeddings_np)  # Add embeddings to the index
    return index

# Example usage
index = create_faiss_index(chunk_embeddings)
def query_to_embedding(query):
    # Convert user query to embedding
    query_embedding = model.encode([query])
    return query_embedding

def retrieve_relevant_chunks(query_embedding, index, k=3):
    # Perform similarity search (find top k closest chunks)
    distances, indices = index.search(np.array(query_embedding).astype(np.float32), k)

    # Retrieve the corresponding chunks based on the indices
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Example usage
query = "What is the latest research at the University of Chicago?"
query_embedding = query_to_embedding(query)
relevant_chunks = retrieve_relevant_chunks(query_embedding, index)
print(relevant_chunks)  # Print the top relevant chunks
def generate_response(query, relevant_chunks):
    # Concatenate retrieved chunks and form a prompt for the response
    context = " ".join(relevant_chunks)
    response = f"Query: {query}\nContext: {context}\nAnswer:"
    return response

# Example usage
response = generate_response(query, relevant_chunks)
print(response)  # Output the response based on the retrieved chunks
