import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re   # Needed for regex cleanup

# Helper function to safely parse JSON from Gemini
def safe_json_parse(text):
    cleaned = text.strip()
    # Remove markdown code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # If the response does not start with '{', extract the JSON block
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    return json.loads(cleaned)

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_structured_data(resume_text):
    prompt = f"""
You are an AI assistant. Read the following resume and extract structured JSON.

Return ONLY valid JSON matching this format. Do not add anything else.

{{
    "skills": ["Skill1", "Skill2"],
    "education": ["Education1", "Education2"],
    "experience": ["Experience1", "Experience2"]
}}

Resume:
\"\"\"
{resume_text}
\"\"\"
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        raw = response.text.strip()
        print("=== Gemini raw response ===")
        print(raw)

        # Use the safe parser instead of json.loads
        return safe_json_parse(raw)

    except json.JSONDecodeError as e:
        return {
            "skills": [],
            "education": [],
            "experience": [],
            "error": f"Invalid JSON returned: {str(e)}. Raw output: {raw}"
        }
    except Exception as e:
        return {
            "skills": [],
            "education": [],
            "experience": [],
            "error": str(e)
        }
  