# Basque Chatbot with RAG and Embeddings

This repository contains code for building a Basque-speaking chatbot powered by Retrieval-Augmented Generation (RAG) with knowledge from movie subtitles. This chatbot is designed to understand and respond in Basque, enhanced with natural language knowledge from dialogues, leveraging the FastAPI framework for web-based interaction.

## Features

- Basque Language Support: The chatbot is trained to interact only in Basque.
- Retrieval-Augmented Generation (RAG): Uses a retrieval system for contextually aware responses.
- Movie Dialogue Database: Enhanced natural language understanding sourced from movie subtitles.
- FastAPI: Provides a simple HTTP API interface for easy integration.

## Prerequisites

- Python 3.10 or 3.11
- Poetry: Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/nicolluis10/kAIxo_X01
   cd kAIxo_01
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   - Rename the `.env.example` file to `.env` and update the variables inside with your own values. Example:

   ```bash
   mv .env .env
   ```


3. Run the server:

   ```bash
   uvicorn main:app --reload
   ```

4. Open localhost:

http://127.0.0.1:8000/


### References

This dataset and publication is a result of the project CONVERSA (TED2021-132470B-I00) funded by MCIN/AEI /10.13039/501100011033 and by "European Union NextGenerationEU/PRTR".