import io
from pypdf import PdfReader

def parse_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file and formats it 
    to match the structure of our Arxiv papers.
    """
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Create a mock metadata structure so it plays nice with the Arxiv data
        return {
            "title": uploaded_file.name,
            "authors": ["User Uploaded"],
            "summary": text[:5000],  # Truncate to avoid token limits if too huge
            "published": "Local File",
            "pdf_url": "#",
            "cluster": -1, # Will be assigned later
            "source": "local"
        }
    except Exception as e:
        return None