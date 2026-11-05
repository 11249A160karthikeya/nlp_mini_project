
# SCSVMV AcademicQuery AI

## Project Features

- AI-powered academic chatbot
- PDF document upload
- Word DOCX document upload
- PowerPoint PPTX document upload
- Excel XLSX document upload
- CSV document upload
- TXT and Markdown upload
- Semantic document search
- Natural-language AI responses
- Follow-up conversations
- Document references with page numbers
- No external website links in intended chatbot responses

## Technologies

- Python
- Streamlit
- OpenAI API
- Sentence Transformers
- PyMuPDF
- NumPy

## Installation

Install the required packages:

pip install -r requirements.txt

## Run Locally

streamlit run app.py

## Render Deployment

Start Command:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Environment Variables

Add these variables in Render:

OPENAI_API_KEY = your_private_api_key

OPENAI_MODEL = gpt-4o-mini

Never upload your API key to GitHub.

## Important Information

Uploaded documents are currently held in
Streamlit session memory.

Different devices have separate sessions.

For permanent shared document storage,
a database and persistent file storage
must be configured.

Scanned PDFs require OCR before
their contents can be searched.
