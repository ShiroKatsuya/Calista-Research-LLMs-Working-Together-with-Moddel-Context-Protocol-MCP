import pdfplumber
import fitz  
import sys
import pytesseract
from PIL import Image
import io
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_content):
    """
    Extract text content from a PDF using PyMuPDF
    
    Args:
        pdf_content (bytes): The binary content of the PDF
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Open PDF from binary content
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Extract text from each page
        text = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
            
            # Check if page has low text content, might be scanned - try OCR if available
            if TESSERACT_AVAILABLE and len(page_text.strip()) < 100:
                try:
                    # Render page to image
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Use OCR to extract text
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text.append(f"\n[OCR Text from Page {page_num + 1}]:\n{ocr_text}")
                except Exception as e:
                    print(f"OCR error on page {page_num + 1}: {e}")
        
        pdf_document.close()
        return "\n".join(text)
        
    except Exception as e:
        # Fallback to pdfplumber if fitz fails
        print(f"PyMuPDF extraction failed, trying pdfplumber: {e}")
        try:
            text = []
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text.append(f"\n--- Page {i + 1} ---\n{page_text}")
            return "\n".join(text)
        except Exception as e2:
            print(f"All PDF extraction methods failed: {e2}")
            return f"[Error extracting PDF content: {e2}]"