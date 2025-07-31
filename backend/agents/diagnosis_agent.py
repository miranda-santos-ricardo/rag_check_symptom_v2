import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DiagnosisAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def suggest_diagnosis(self, matches):
        if not matches:
            return "Sorry, no matching symptoms were found."

        context = "\n".join([f"{doc}" for doc, _ in matches])
        prompt = f"""You are a medical assistant AI.
Based on the following symptom descriptions and possible disease matches:

{context}

Please provide a short diagnostic summary of what conditions the user might have, and why, based on the symptoms."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful and concise medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()
