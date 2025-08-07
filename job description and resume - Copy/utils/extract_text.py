import os
import google.generativeai as genai
import json
import PyPDF2
import docx2txt

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_structured_data(resume_text):
    prompt = f"""
You are an intelligent resume parser.

**IMPORTANT:**
Return JSON only. Do NOT add any explanations, text, or markdown code fences.

Format example:
{{
  "skills": ["skill1", "skill2"],
  "education": ["degree1", "degree2"],
  "experience": ["experience1", "experience2"]
}}

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Use the safe parser
        return safe_json_parse(raw)

    except Exception as e:
        return {"skills": [], "education": [], "experience": [], "error": str(e)}

def safe_json_parse(raw):
    try:
        return json.loads(raw)
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}"}

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        text = ''
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        return text
    elif ext in ['.docx', '.doc']:
        return docx2txt.process(file_path)
    else:
        raise ValueError('Unsupported file type: ' + ext)
