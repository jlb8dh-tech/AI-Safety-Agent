import google.generativeai as genai
import os
from dotenv import load_dotenv

# I put my API key in another file 
load_dotenv()

# https://ai.google.dev/api/generate-content

try: genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: Something went wrong with the API Key.\n")
    exit()

model = genai.GenerativeModel('models/gemini-pro-latest')

# configures the style of the output from Gemini. I can tweak these settings
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=1.0,
    top_k=1
)

print("Please enter a prompt \n")
prompt = input("Please enter your training module request (e.g., Create a 5-minute forklift safety training module):\n> ")

print("--- Generating Outline ---")
outline_prompt = f"""
You are an expert instructional designer specializing in manufacturing and industrial training.
Your task is to create a structured outline for an e-learning module based on the user request.
The content must be relevant for a manufacturing audience.

USER REQUEST: "{prompt}"

Generate a structured outline for this module. The outline must include:
1.  A main title for the course.
2.  A list of 2-3 specific learning objectives (e.g., "Identify 3 primary pinch points on the equipment," "List the required PPE based on OSHA guidelines").
3.  A list of sections or modules (e.g., "Section 1: Introduction to [Equipment]", "Section 2: Pre-Operation Safety Checks").

Respond only with the outline.
"""

response_outline = model.generate_content(outline_prompt, generation_config = generation_config)
course_outline = response_outline.text
print(course_outline)


print("\n--- Generating Lesson Text for Each Section ---")
lesson_prompt = f"""
You are an e-learning content writer for industrial and manufacturing training.
Your task is to write the narration/lesson text based on the provided outline.
The tone must be clear, direct, and professional, prioritizing safety and compliance.
Reference manufacturing concepts like equipment operation, safety protocols, and OSHA standards where appropriate.

USER REQUEST: "{prompt}"

COURSE OUTLINE:
{course_outline}

Write the detailed lesson text for each section in the outline.
This text will be used for narration, so write it to be spoken.
Ensure the content directly teaches the learning objectives from the outline.

Respond only with the lesson text, formatted clearly by section.
"""

response_lesson = model.generate_content(lesson_prompt, generation_config=generation_config)
lesson_text = response_lesson.text
print(lesson_text)


print("\n--- Generating Quiz ---")
quiz_prompt = f"""
You are a quiz creator for corporate compliance and safety training.
Your task is to create a short quiz based *only* on the provided manufacturing lesson text.
The questions must be practical and test for understanding of the safety and operational procedures.

LESSON TEXT:
{lesson_text}

Generate 3-5 quiz questions.
- The questions should be multiple-choice or true/false.
- The questions must be directly answerable from the lesson text.
- For each question, clearly indicate the correct answer.

Respond only with the quiz questions and their answers.
"""

response_quiz = model.generate_content(quiz_prompt, generation_config=generation_config)
quiz_content = response_quiz.text
print(quiz_content)