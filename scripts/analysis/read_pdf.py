import sys
try:
    import pypdf
    reader = pypdf.PdfReader('/data/qbao775/Explanation-Generation/35164-Article Text-39231-1-2-20250410.pdf')
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open('/data/qbao775/Explanation-Generation/temp_pdf.txt', 'w') as f:
        f.write(text)
    print("Success with pypdf")
except Exception as e:
    print(f"Error: {e}")
