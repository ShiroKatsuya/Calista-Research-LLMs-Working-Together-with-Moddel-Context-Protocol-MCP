import re
import textwrap

def format_content(content, score_map=None):

    content = content.replace('\r\n', '\n').strip()
    
    # Remove extra whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove very short lines (likely menu items, breadcrumbs, etc.)
    content = re.sub(r'(?m)^.{1,20}$\n', '', content)
    
    # Remove any remaining header/footer-like text
    content = re.sub(r'(?i)^(home|menu|navigation|contact us|about us|privacy policy|terms of service)[\s\n]*$', '', content, flags=re.MULTILINE)
    
    # Enhanced formatting
    paragraphs = re.split(r'\n\s*\n', content)
    formatted_paragraphs = []
    
    horizontal_line = "─" * 80
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # Get relevance score for this paragraph if available
        relevance_score = score_map.get(para.strip(), 0) if score_map else 0
        high_relevance = relevance_score >= 0.25  # Threshold for high relevance highlighting
            
        # Format headers with proper styling
        if para.startswith('###'):
            header_text = para.strip('#').strip()
            formatted_paragraphs.append(f"\n{horizontal_line}")
            formatted_paragraphs.append(f"📌 {header_text.upper()} 📌")
            formatted_paragraphs.append(f"{horizontal_line}")
            continue
        
        # Format subheaders (detect potential subheadings)
        elif re.match(r'^[A-Z][A-Za-z\s]{5,50}:$', para.strip()) or para.strip().isupper():
            formatted_paragraphs.append(f"▶ {para.strip()} ◀")
            continue
            
        # Improve list formatting
        elif para.strip().startswith('•') or re.match(r'^\s*[\d\-\*]+\.?\s', para.strip()):
            list_items = re.split(r'\n', para.strip())
            for item in list_items:
                # Standardize list bullets and add indentation
                item = re.sub(r'^\s*[\d\-\*]+\.?\s', '• ', item.strip())
                if not item.startswith('•'):
                    item = f"• {item}"
                    
                # Highlight highly relevant list items
                if high_relevance:
                    formatted_paragraphs.append(f"  🔍 {item}")
                else:
                    formatted_paragraphs.append(f"  {item}")
            continue
            
        # Enhance quote formatting
        elif para.strip().startswith('>'):
            quote_text = para.strip('> ').strip()
            wrapped_quote = textwrap.fill(quote_text, width=76)
            quote_lines = wrapped_quote.split('\n')
            formatted_quote = ['┌' + '─' * 78 + '┐']
            for line in quote_lines:
                formatted_quote.append(f"│ \"{line}\"" + ' ' * (76 - len(line)) + ' │')
            formatted_quote.append('└' + '─' * 78 + '┘')
            formatted_paragraphs.append('\n'.join(formatted_quote))
            continue
            
        # Highlight important information (keywords)
        important_keywords = ['important', 'note', 'warning', 'caution', 'key', 'critical', 'essential']
        has_important = any(keyword in para.lower() for keyword in important_keywords)
        
        # Format normal paragraphs
        single_line = re.sub(r'\s+', ' ', para.strip())
        
        # Add emphasis based on importance and relevance
        if has_important:
            wrapped = textwrap.fill(single_line, width=76)
            formatted_paragraphs.append(f"❗ {wrapped} ❗")
        elif high_relevance:
            # Highlight highly relevant content with a magnifying glass
            wrapped = textwrap.fill(single_line, width=75)
            formatted_paragraphs.append(f"🔍 {wrapped}")
        else:
            wrapped = textwrap.fill(single_line, width=80)
            formatted_paragraphs.append(wrapped)
    
    # Add proper spacing
    formatted_text = '\n\n'.join(formatted_paragraphs)
    
    # Format URLs in text for better visibility
    formatted_text = re.sub(r'(https?://\S+)', r'🔗 \1', formatted_text)
    
    return formatted_text