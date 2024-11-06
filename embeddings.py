import os
import json
import time

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables for authentication
load_dotenv()

# Define the directory containing the JSONL files and the persistent directory for the Chroma DB
current_dir = os.path.dirname(os.path.abspath(__file__))
jsonl_dir = os.path.join(current_dir, "open_subtitles_eu_all")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Function to load JSONL files one at a time
def process_jsonl_file(file_path):
    dialogues = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "dialogues" in item:
                dialogues.extend(item["dialogues"])
    return dialogues


# Function to approximate token counting based on words
def count_tokens(text):
    return len(text.split())


# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Updated to a valid embedding model
    )
    print("\n--- Finished creating embeddings ---")

    # Token limits and batching
    max_tokens_per_minute = 1000000  # Your actual limit
    batch_token_limit = 250000  # Reduce further to avoid exceeding the limit
    current_token_count = 0
    batch = []

    # Iterate over JSONL files one at a time
    for root, dirs, files in os.walk(jsonl_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Process the dialogues in the current JSONL file
                dialogues = process_jsonl_file(file_path)

                # Concatenate the dialogues and split them into chunks
                concatenated_dialogues = "\n".join(dialogues)
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_text(concatenated_dialogues)

                # Process document chunks and add them to the batch
                for doc in docs:
                    tokens_in_doc = count_tokens(doc)

                    # Check if adding this document would exceed the token limit
                    if current_token_count + tokens_in_doc > batch_token_limit:
                        print(f"\n--- Processing batch with {current_token_count} tokens ---")
                        db = Chroma.from_texts(batch, embeddings, persist_directory=persistent_directory)

                        # Reset for the next batch
                        batch = []
                        current_token_count = 0

                        # Wait to avoid rate limit, then continue
                        print("Waiting 65 seconds to avoid rate limit...")
                        time.sleep(65)

                    # Add the current document to the batch and update token count
                    batch.append(doc)
                    current_token_count += tokens_in_doc

    # Process any remaining documents in the final batch
    if batch:
        print(f"\n--- Processing final batch with {current_token_count} tokens ---")
        db = Chroma.from_texts(batch, embeddings, persist_directory=persistent_directory)

    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
