from flask import Flask, request, render_template
import os
import hashlib
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.extract_text import extract_text
from utils.parser import extract_structured_data
import PyPDF2 as pdf
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
HASH_FILE = 'data/resume_hashes.json'

# Load Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load hash database
def load_hashes():
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

# Save updated hash database
def save_hashes(hashes):
    with open(HASH_FILE, 'w') as f:
        json.dump(hashes, f)

# Get file hash
def file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Generate AI feedback using Gemini
def get_ai_feedback(resume_text, job_description):
    prompt = f"""
    Act as a professional resume reviewer.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Provide feedback on how the resume can be improved to better match the job description. List missing skills, improvements, and red flags.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # You can also try "gemini-1.5-flash" for faster responses
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

@app.route("/")
def index():
    return render_template('matchresume.html')

@app.route("/matcher", methods=['GET', 'POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')

        if not job_description.strip() or not resume_files:
            return render_template('matchresume.html', message="Please provide job description and resumes.")

        existing_hashes = load_hashes()
        resumes, filenames, feedbacks, parsed_data = [], [], [], []

        for resume_file in resume_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filepath)
            r_hash = file_hash(filepath)

            if r_hash in existing_hashes:
                continue

            text = extract_text(filepath)
            resumes.append(text)
            filenames.append(resume_file.filename)
            existing_hashes[r_hash] = resume_file.filename

            feedback = get_ai_feedback(text, job_description)
            feedbacks.append(feedback)

            structured = extract_structured_data(text)
            parsed_data.append(structured)

        save_hashes(existing_hashes)

        if not resumes:
            return render_template('matchresume.html', message="All resumes were duplicates or unreadable.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        job_vector, resume_vectors = vectors[0], vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        top_indices = similarities.argsort()[-3:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]
        top_feedbacks = [feedbacks[i] for i in top_indices]
        top_structured = [parsed_data[i] for i in top_indices]

        return render_template('matchresume.html',
                               message="Top matching resumes:",
                               top_resumes=top_resumes,
                               similarity_scores=similarity_scores,
                               ai_feedbacks=top_feedbacks,
                               structured_data=top_structured,
                               zip=zip)

    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
