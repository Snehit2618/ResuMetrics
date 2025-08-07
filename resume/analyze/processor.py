import re
import PyPDF2
import spacy
import joblib
import cohere
from googleapiclient.discovery import build
from concurrent.futures import ThreadPoolExecutor

# Load resources once at startup
nlp = spacy.load("en_core_web_sm")
model = joblib.load("analyze/utils/model.pkl")
vectorizer = joblib.load("analyze/utils/vectorizer.pkl")

# API key for YouTube Data API
api_key = "AIzaSyASNQF4YtN1KkyB7XIzp4ljtgRCk_xMhIw"

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def extract_skills(text, skills_set):
    text_lower = text.lower()
    return {skill for skill in skills_set if skill in text_lower}

def predict_role(resume_text, skills_list):
    resume_skills = " ".join(extract_skills(resume_text, skills_list))
    resume_vectorized = vectorizer.transform([resume_skills])
    predicted_role = model.predict(resume_vectorized)[0]
    return predicted_role

def provide_feedback_with_cohere(missing_skills, role):
    co = cohere.Client('KbGRbPUqMxBrDJF58JrwdBZvwqbngYaSs8VFNTSw')
    missing_skills_str = ", ".join(missing_skills)
    prompt = f"""
    The candidate is applying for the role of '{role}'. Their resume is missing the following skills: {missing_skills_str}.
    Provide detailed suggestions on how the candidate can improve their resume, considering the role they are applying for.
    """
    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    feedback = response.generations[0].text.strip()
    return feedback

def recommend_youtube_videos(missing_skills, api_key):
    """
    Fetch YouTube video recommendations for missing skills.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    recommendations = {}

    for skill in missing_skills:
        request = youtube.search().list(
            part="snippet",
            q=f"{skill} tutorial",
            type="video",
            maxResults=5
        )
        response = request.execute()

        videos = []
        for item in response['items']:
            title = item['snippet']['title']
            video_id = item['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append(f"{title}: {video_url}")

        recommendations[skill] = videos

    return recommendations

def process_resume(resume_path, jd_text, skills_set):
    """
    Process a resume to predict the role, match skills, generate feedback, and recommend resources.
    """
    # Extract text from the resume
    resume_text = extract_text_from_pdf(resume_path)

    # Predict the role
    predicted_role = predict_role(resume_text, skills_set)

    # Extract skills from both resume and job description
    resume_skills = extract_skills(resume_text, skills_set)
    jd_skills = extract_skills(jd_text, skills_set)

    # Determine matching and mismatched (missing) skills
    matching_skills = resume_skills.intersection(jd_skills)
    missing_skills = jd_skills.difference(resume_skills)

    # Generate feedback for the missing skills
    feedback = provide_feedback_with_cohere(missing_skills, predicted_role)

    # Fetch YouTube video recommendations for missing skills
    video_recommendations = recommend_youtube_videos(missing_skills, api_key)

    # Create result dictionary
    result = {
        "predicted_role": predicted_role,
        "match_percentage": len(matching_skills) / len(jd_skills) * 100 if jd_skills else 0,
        "matching_skills": list(matching_skills),  # Convert to list for JSON serialization
        "missing_skills": list(missing_skills),   # Convert to list for JSON serialization
        "feedback": feedback,
        "video_recommendations": video_recommendations,
        "status": "Shortlisted ✅" if len(matching_skills) / len(jd_skills) * 100 >= 70 else "Not Shortlisted ❌"
    }

    return result