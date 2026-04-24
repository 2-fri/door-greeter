from groq import Groq
from dotenv import load_dotenv
import os

# Threading
import threading

# Speech To Text
import speech_recognition as sr

# Text to Speech
import pyttsx3
import os

SYSTEM_PROMPT = """
You are a friendly and helpful greeter bot stationed at the entrance of a building. 
Your job is to welcome people as they enter and say goodbye as they leave. 
System messages will be sent to you in the format "Person {id} ENTERED the frame. Person {id} description: {description}" or "Person {id} LEFT the frame."
Use these messages to keep track of who is currently in front of you and to generate appropriate greetings and goodbyes.
The description field might be empty, if so, attempt to request the person's name and use it. Do NOT refer to people by their ID.
If you believe that the appropriate response is staying quiet, leave the response blank.
The speech recognition system can make mistakes, ask for clarification if the user seems to be saying something odd.
Do not pot emojis or emoticons in your responses.
"""

SUMMARY_PROMPT = "Your job is to concisely (500 characters max) summarize all important information about Person "

class Converser:
    n_people = 0
    conversation = None

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.recognizer = sr.Recognizer()
        self.state = []
        self.tts_engine = pyttsx3.init()

    def add_person(self, id : int, description : str):
        self.state.append({"role": "system", "content": f"Person {id} ENTERED the frame. Person {id} description: {description}"})
        self.n_people += 1
        if (self.n_people == 1): # Start loop if first person enters
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.state
            )
            self.state.append({"role": "assistant", "content": completion.choices[0].message.content})
            print(f"ROBOTENTRY> {self.state[-1]['content']}")
            self.speak(self.state[-1]['content'])
            self.conversation = threading.Thread(target=self.conversation_loop, daemon=False)
            self.conversation.start()


    def remove_person(self, id : int):
        self.n_people -= 1
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SUMMARY_PROMPT + str(id)}] + self.state
        )
        self.state.append({"role": "system", "content": f"Person {id} LEFT the frame."})
        if (self.n_people <= 0): # Reset state if conversation over (everybody left)
            self.n_people = 0
            self.conversation.join()
            self.state = []
            self.conversation = None
        return completion.choices[0].message.content

    # Listens to user input and returns it as text
    def listen(self):
        with sr.Microphone() as source:
            print("Listening for user response...")
            audio = self.recognizer.listen(source, phrase_time_limit = 10)
        try:
            user_message = self.recognizer.recognize_faster_whisper(audio)
            print("USER> " + user_message)
            return user_message
        except sr.UnknownValueError:
            print("Recognizer could not understand audio")
        except sr.RequestError as e:
            print("Recognizer error; {0}".format(e))
    

    # Take input from the user and produce a responce
    def respond(self, message):
        if (message.strip() == ""):
            return # Empty Input
        self.state.append({"role": "user", "content": message})
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.state
        )
        self.state.append({"role": "assistant", "content": completion.choices[0].message.content})
        # Speak
        print(f"ROBOTCONVO> {self.state[-1]['content']}")
        self.speak(self.state[-1]['content'])
    
    # Get text and pronounce it with TTS
    def speak(self, message):
        if (message.strip() == ""):
            return # Empty Message
        try: # Attempt to use the high quality TTS
            voice = self.client.audio.speech.create(
                model = "canopylabs/orpheus-v1-english",
                voice = "troy",
                input = message,
                response_format = "wav"
            )
            voice.write_to_file("output.wav")
            os.system("aplay output.wav")
        except: # Free fallback
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()

    # Run this in a thread, Keep the conversation going
    def conversation_loop(self):
        while self.n_people > 0:
            self.respond(self.listen())
