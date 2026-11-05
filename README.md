🧠 Multimodal RAG & Document Intelligence Studio
🚀 Overview

Multimodal RAG & Document Intelligence Studio is an AI-powered application that enables users to interact intelligently with PDF documents containing both textual and visual information.

Traditional Retrieval-Augmented Generation (RAG) systems primarily focus on extracting and retrieving text. However, technical documents often contain important information in diagrams, charts, screenshots, illustrations, and other visual elements.

This project addresses that limitation by combining text-based retrieval and image-based retrieval into a unified multimodal pipeline.

Users can upload a PDF, ask natural-language questions, and receive AI-generated responses using relevant textual and visual context retrieved from the document.
✨ Key Features

    PDF Document Processing: Extracts textual content and visual elements from uploaded PDF documents.
    Text Retrieval: Uses Sentence Transformers to generate semantic embeddings for document text.
    Image Retrieval: Uses CLIP embeddings to represent document images in a searchable vector space.
    Multimodal Search: Retrieves relevant information from both text and images.
    Vector Similarity Search: Uses FAISS for efficient similarity-based retrieval.
    AI-Powered Responses: Uses the Groq API to generate contextual responses using retrieved information.
    Document Question Answering: Allows users to ask questions about the contents of uploaded documents.
    Visual Context Retrieval: Helps retrieve relevant diagrams, screenshots, and other visual content.
    Interactive Interface: Provides a user-friendly interface built with Streamlit.

🏗️ System Architecture

The application follows a multimodal Retrieval-Augmented Generation pipeline.

                 ┌─────────────────────────┐
                 │       User Uploads PDF  │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    PDF Document Parser  │
                 │       (PyMuPDF)         │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │   Text Extraction│      │  Image Extraction│
       └────────┬─────────┘      └────────┬─────────┘
                │                         │
                ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │ Sentence         │      │ CLIP Image       │
       │ Transformer      │      │ Embeddings       │
       │ Embeddings       │      │                  │
       └────────┬─────────┘      └────────┬─────────┘
                │                         │
                ▼                         ▼
       ┌──────────────────┐      ┌──────────────────┐
       │ Text FAISS Index │      │ Image FAISS Index│
       └────────┬─────────┘      └────────┬─────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │     User Query          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Multimodal Retrieval  │
                 │   Text + Image Search   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   Retrieved Context    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      Groq LLM API       │
                 │   Contextual Generation│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   AI-Generated Answer   │
                 └─────────────────────────┘

🛠️ Technology Stack
Category 	Technologies
Programming Language 	Python
Frontend 	Streamlit
PDF Processing 	PyMuPDF
Text Embeddings 	Sentence Transformers
Image Embeddings 	CLIP (clip-ViT-B-32)
Vector Database 	FAISS
LLM Integration 	Groq API
Numerical Computing 	NumPy
Image Processing 	Pillow
Deployment 	Render
Version Control 	Git & GitHub
🧩 How It Works
1. PDF Ingestion

The user uploads a PDF document through the Streamlit interface.

PyMuPDF processes the document and extracts:

    Textual content.
    Embedded images.
    Page-level information.

2. Text Embedding Generation

The extracted text is divided into smaller chunks.

Sentence Transformers converts these chunks into numerical vector representations using a semantic embedding model.

These embeddings are stored in a FAISS text index.
3. Image Embedding Generation

The document's visual content is processed using the CLIP model.

CLIP converts images into vector representations that capture their semantic information.

These embeddings are stored separately in a FAISS image index.
4. Multimodal Retrieval

When a user submits a question, the application performs semantic retrieval to identify relevant information from the available text and image representations.

FAISS is used to find similar vector representations efficiently.
5. Response Generation

The retrieved context is passed to the Groq API.

The language model uses the retrieved information to generate a contextual response to the user's question.
6. Display Results

The application presents the generated answer through the Streamlit interface, along with relevant retrieved document content where supported by the application.
📂 Project Structure

multimodal-rag-vision/
│
├── app.py
├── core_services.py
├── requirements.txt
│
├── project/
│   ├── README.md
│   │
│   ├── images/
│   │
│   ├── pdfs/
│   │
│   ├── vector_db/
│   │   ├── text.index
│   │   ├── image.index
│   │   ├── metadata.db
│   │   └── metadata_store.json
│   │
│   ├── text_chunks.json
│   ├── images.json
│   ├── text.index
│   └── image.index
│
└── ...

Note: The structure above highlights the main application and project assets. Some generated files may be stored at the repository root depending on the indexing workflow.
⚙️ Installation & Setup

Follow these steps to run the application locally.
Prerequisites

    Python 3.10 or later
    Git
    A Groq API key

1. Clone the Repository

git clone https://github.com/mohammadshaikh77/RAG---ADVANCE-DOCUMENT-INTELLIGENCE.git

2. Navigate to the Project Directory

cd RAG---ADVANCE-DOCUMENT-INTELLIGENCE

3. Create a Virtual Environment

python -m venv venv

Activate it:

macOS / Linux

source venv/bin/activate

Windows

venv\Scripts�ctivate

4. Install Dependencies

pip install -r requirements.txt

5. Configure Environment Variables

Create a .env file if your application uses environment-based configuration.

GROQ_API_KEY=your_groq_api_key

Get your API key from:

https://console.groq.com/keys

Important: Never commit your actual API key to GitHub. Configure it securely using environment variables.
6. Run the Application

streamlit run app.py

The application will be available at:

http://localhost:8501

🌐 Live Deployment

The application is deployed using Render.

🔗 Live Application:
https://multimodal-rag-vision.onrender.com

The application can be accessed directly through the public URL without requiring a local installation.

Note: The free hosting instance may take some time to respond after a period of inactivity.
🔑 Environment Variables
Variable 	Description
GROQ_API_KEY 	API key used for LLM integration

Keep API credentials private and configure them through your deployment platform's environment settings.
🎯 Use Cases

This project can be useful for:

    Technical Documentation: Understanding technical PDFs containing diagrams and explanations.
    Academic Research: Extracting information from research papers and study materials.
    Engineering Documents: Retrieving information from architecture diagrams and technical illustrations.
    Business Reports: Interacting with reports containing charts, tables, and textual information.
    Educational Content: Asking questions about notes, textbooks, and learning materials.

🔮 Future Improvements

Potential improvements include:

    Support for multiple PDF documents and document collections.
    Advanced hybrid retrieval combining keyword and semantic search.
    Improved visual understanding using vision-language models.
    Retrieval evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank.
    RAG evaluation using faithfulness and answer relevance metrics.
    Conversation history and persistent user sessions.
    Improved document citation and page-level source references.
    Optimized embedding generation and caching.
    Support for additional document formats.

👨‍💻 Author

Mohammad Shaikh

Information Technology Student
Dwarkadas J. Sanghvi College of Engineering, Mumbai
