# RAG Project: Retrieval-Augmented Generation Pipeline

# RAG Project

A Retrieval-Augmented Generation (RAG) pipeline for uploading documents, querying their content, and viewing metadata. Built for the LLM Specialist Assignment.

## Overview
This project implements a RAG system using FastAPI, FAISS (vector database), and the Gemini API (`gemini-1.5-flash`). Users can upload documents (PDFs or text), query their content, and view metadata. The app is containerized with Docker for local and cloud deployment.

## Requirements
- Python 3.10+
- Docker (for containerized deployment)
- Gemini API key (free tier) from [ai.google.dev](https://ai.google.dev)

## Setup and Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RAG_project.git
   cd RAG_project
   
## Project Structure

```plaintext
rag_project/
├── README.md             # Project overview and documentation
├── requirements.txt      # Python package dependencies
├── .gitignore            # Files and directories to ignore in version control
├── setup.py              # (Optional) Packaging script for the project
├── src/
│   ├── __init__.py       # Marks src as a Python package
│   └── main.py           # Main script containing the RAG pipeline code
├── articles/             # Directory containing input document files
│   └── sample_article.txt  # Example article (add your own files here)
└── notebooks/            # (Optional) Jupyter notebooks for experiments and demos
    └── example.ipynb
