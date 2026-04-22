from groq import Groq
from dotenv import load_dotenv
import os

SYSTEM_PROMPT = """
You are a friendly and helpful greeter bot stationed at the entrance of a building. 
Your job is to welcome people as they enter and say goodbye as they leave. 
System messages will be sent to you in the format "Person {id} ENTERED the frame. Person {id} description: {description}" or "Person {id} LEFT the frame."
Use these messages to keep track of who is currently in front of you and to generate appropriate greetings and goodbyes.
The description field might be empty, if so, attempt to request the person's name and use it. Do NOT refer to people by their ID.
If you believe that the appropriate response is staying quiet, leave the response blank.
"""

SUMMARY_PROMPT = """
Please provide a 300 character MAXIMUM summary of all the information that should be remembered about ONLY the person that just left.
Make sure to include their name if it was mentioned, but omit any particularly sensitive information or information that the user asked not to remember.
"""

class Converser:
    n_people = 0

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.state = []

    def add_person(self, id : int, description : str):
        self.state.append({"role": "system", "content": f"Person {id} ENTERED the frame. Person {id} description: {description}"})
        self.n_people += 1

    def remove_person(self, id : int):
        self.state.append({"role": "system", "content": f"Person {id} LEFT the frame."})
        self.n_people -= 1
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.state + [{"role": "system", "content": SUMMARY_PROMPT}]
        )
        if (self.n_people <= 0): # Reset state if conversation over (everybody left)
            self.n_people = 0
            self.state = []
        return completion.choices[0].message.content
    
    def converse(self, message):
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.state + [{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content