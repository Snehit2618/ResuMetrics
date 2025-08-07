
from django.conf import settings
from django.shortcuts import render, HttpResponse
from .form import ResumeUploadForm
import os
import pickle
import docx
import PyPDF2
import re



def index2(request):
    return render(request,'home2.html')

def index1(request):
    return render(request,'home1.html')

def index(request):
    return render(request,'home.html')
# Load pre-trained model, TF-IDF vectorizer, and label encoder from the static directory

import os
import re
import pickle
import PyPDF2
import docx
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render
from io import BytesIO

# Loading the models from the static folder (these use filesystem paths, not in-memory files)
import os
import re
import pickle
import PyPDF2
import docx
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render
from io import BytesIO

# Load models from static folder
model_path = os.path.join(settings.STATICFILES_DIRS[0], 'models')
svc_model = pickle.load(open(os.path.join(model_path, 'clf.pkl'), 'rb'))
tfidf = pickle.load(open(os.path.join(model_path, 'tfidf.pkl'), 'rb'))
le = pickle.load(open(os.path.join(model_path, 'encoder.pkl'), 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)
    try:
        # PyPDF2 expects a file-like object; using uploaded_file.read() directly
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
        return text
    except Exception as e:
        raise ValueError(f"Error extracting PDF: {str(e)}")

# Extract text from DOCX
def extract_text_from_docx(uploaded_file):
    uploaded_file.seek(0)
    try:
        doc = docx.Document(uploaded_file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        raise ValueError(f"Error extracting DOCX: {str(e)}")

# Extract text from TXT
def extract_text_from_txt(uploaded_file):
    uploaded_file.seek(0)
    try:
        text = uploaded_file.read().decode('utf-8')
        return text
    except UnicodeDecodeError:
        text = uploaded_file.read().decode('latin-1')
        return text
    except Exception as e:
        raise ValueError(f"Error extracting TXT: {str(e)}")

# Handle file upload and text extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Predict category based on resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Django view to handle resume uploads
def upload_resume(request):
    if request.method == 'POST':
        form = ResumeUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES.get('file')
            
            if uploaded_file:
                try:
                    # Extract text from the uploaded resume
                    resume_text = handle_file_upload(uploaded_file)
                    
                    # Make prediction based on extracted text
                    predicted_category = pred(resume_text)

                    # Render template with predicted category and resume text
                    return render(request, 'upload_resume.html', {
                        'form': form,
                        'predicted_category': predicted_category,
                        'resume_text': resume_text,  # Optionally display the extracted text
                    })
                except Exception as e:
                    # Handle errors during file processing
                    return render(request, 'upload_resume.html', {
                        'form': form,
                        'error': f"Error during processing: {str(e)}",
                    })
            else:
                # Handle case where no file is uploaded
                return render(request, 'upload_resume.html', {
                    'form': form,
                    'error': "No file uploaded. Please upload a file.",
                })
    else:
        form = ResumeUploadForm()

    return render(request, 'upload_resume.html', {'form': form})





from django.shortcuts import render
from .processor import process_resume
import os

import os
from django.conf import settings


def upload_view(request):
    context = {"results": None}  # Initialize context with no results

    if request.method == "POST":
        resumes = request.FILES.getlist("resumes")
        jd_file = request.FILES["jd"]
        skills_file = request.FILES["skills"]

        # Read job description and skills file
        jd_text = jd_file.read().decode("utf-8")
        skills_set = set(skills_file.read().decode("utf-8").splitlines())

        results = []
        for resume in resumes:
            resume_path = f"temp_{resume.name}"
            with open(resume_path, "wb") as temp_file:
                temp_file.write(resume.read())
            
            # Process the resume
            result = process_resume(resume_path, jd_text, skills_set)
            results.append(result)

            # Clean up temporary file
            os.remove(resume_path)

        # Add results to the context
        context["results"] = results

    return render(request, "upload_and_results.html",context)



from django.http import JsonResponse
from django.shortcuts import render
import google.generativeai as genai
import asyncio
import re

# Configure the API key
genai.configure(api_key="AIzaSyBAQSudWzi7MYBo78VQ2dEW6Rpf-v_FIRk")

# =========================
# Chatbot Functions
# =========================

async def generate_question(skill):
    """Generates a technical interview question for a given skill."""
    prompt = f"Generate a technical interview question for a candidate proficient in {skill}. Only provide the question."
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating question: {str(e)}"


async def evaluate_answer(candidate_answer, question):
    """Evaluates the candidate's answer and returns a score out of 5."""
    prompt = (
        f"Evaluate the following answer: '{candidate_answer}' to the question: "
        f"'{question}'. Provide ONLY a numerical score out of 5, without explanation."
    )
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    
    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        score = extract_score(response.text)
        return score
    except Exception as e:
        return 0  # Default score if there's an issue


def extract_score(response_text):
    """Extracts a numerical score from the response text."""
    match = re.search(r"\b([0-5])\b", response_text)  # Matches numbers 0-5
    return int(match.group(1)) if match else 0


# =========================
# Django Views
# =========================

def interview_view(request):
    """Handles the interview process by generating and scoring questions."""
    if request.method == 'POST':
        candidate_name = request.POST.get('candidate_name', 'Candidate')
        skills_input = request.POST.get('skills', '')

        skills = [skill.strip() for skill in skills_input.split(",") if skill.strip()]

        if not skills:
            return JsonResponse({'error': 'No skills provided. Please enter at least one skill.'}, status=400)

        # Run the interview process asynchronously
        questions_and_scores = asyncio.run(start_interview(skills))

        return JsonResponse(questions_and_scores, safe=False)

    return render(request, 'interview.html')


async def start_interview(skills):
    """Runs an interview session with 5 questions, evaluating each answer."""
    total_score = 0
    num_questions = 5
    questions_and_scores = []

    for i in range(num_questions):
        skill = skills[i % len(skills)]  # Rotate through skills
        question = await generate_question(skill)

        # Placeholder for actual user input handling
        candidate_answer = "Simulated answer"  # Replace this with user input in production

        score = await evaluate_answer(candidate_answer, question)
        total_score += score

        questions_and_scores.append({
            "question": question,
            "score": score
        })

    return {
        "questions": questions_and_scores,
        "total_score": total_score,
        "max_score": num_questions * 5
    }


async def evaluate_answer_view(request):
    """Evaluates a candidate's answer to a specific question."""
    if request.method == 'POST':
        candidate_answer = request.POST.get('answer', '')
        question = request.POST.get('question', '')

        if not candidate_answer or not question:
            return JsonResponse({'error': 'Both question and answer are required.'}, status=400)

        score = await evaluate_answer(candidate_answer, question)

        return JsonResponse({"score": score})



import os
from django.shortcuts import render
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.storage import default_storage
import docx2txt
import PyPDF2

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def match_resume(request):
    if request.method == 'POST':
        # Get the job description file
        job_description_file = request.FILES.get('job_description_file')
        if job_description_file:
            # Save and read the job description file
            job_description_filename = default_storage.save(os.path.join('uploads', job_description_file.name), job_description_file)
            job_description_path = os.path.join(settings.MEDIA_ROOT, job_description_filename)
            job_description = extract_text_from_txt(job_description_path)
        else:
            return render(request, 'matchresume.html', {'message': "Please upload a job description file."})

        # Get the resumes
        resume_files = request.FILES.getlist('resumes')
        resumes = []
        filenames = []
        for resume_file in resume_files:
            filename = default_storage.save(os.path.join('uploads', resume_file.name), resume_file)
            file_path = os.path.join(settings.MEDIA_ROOT, filename)
            filenames.append(resume_file.name)
            resumes.append(extract_text(file_path))

        if not resumes:
            return render(request, 'matchresume.html', {'message': "Please upload resumes."})

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 resumes and their similarity scores
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        return render(request, 'matchresume.html', {
            'message': "Top matching resumes:",
            'top_resumes': zip(top_resumes, similarity_scores)
        })

    return render(request, 'matchresume.html')
