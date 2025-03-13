from urllib.parse import urlparse, unquote
def detect_document_type(url, response=None):

    parsed_url = urlparse(url)
    path = unquote(parsed_url.path.lower())
    
    if path.endswith('.pdf'):
        return 'pdf'
    elif path.endswith('.docx'):
        return 'docx'
    elif path.endswith('.doc'):
        return 'doc'
    elif path.endswith('.txt'):
        return 'txt'
    elif path.endswith('.rtf'):
        return 'rtf'
    elif any(path.endswith(ext) for ext in ['.ppt', '.pptx']):
        return 'presentation'
    elif any(path.endswith(ext) for ext in ['.xls', '.xlsx', '.csv']):
        return 'spreadsheet'
    
    # Check content-type header if response is provided
    if response and 'Content-Type' in response.headers:
        content_type = response.headers['Content-Type'].lower()
        if 'application/pdf' in content_type:
            return 'pdf'
        elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            return 'docx'
        elif 'application/msword' in content_type:
            return 'doc'
        elif 'text/plain' in content_type:
            return 'txt'
        elif 'text/html' in content_type:
            return 'html'
        elif 'application/rtf' in content_type:
            return 'rtf'
    

    return 'html'