from groq import Groq
import os

# Threading
import threading

# Speech To Text
import speech_recognition as sr

# Text to Speech
import pyttsx3
import os
from time import sleep
from datetime import datetime
import wave

SYSTEM_PRIMER = """
You are a friendly and helpful greeter bot stationed at the entrance of a building. 
Your job is to welcome people as they enter and say goodbye as they leave.
After the initial greeting, attempt to make small talk and ask why they're here, etc.
The system message immediately after this describes the location you are at and the one after that gives you the current date and time.
After that, there will be some amount of system messages giving information about people that participated or are participating in the current conversation, in the format "Person {id} description: {description}."
System messages regarding who is in the frame will be sent to you in the format "Person {id} ENTERED the frame." or "Person {id} LEFT the frame."
Use these messages to keep track of who is currently in front of you and to generate appropriate greetings and goodbyes.
If you have any information already stored about the user (e.g. where they were the last time they visit), feel free to ask them about it. 
The description field might be empty, if so, attempt to request the person's name and use it. Do NOT refer to people by their ID.
If you already know the person's name, don't ask for it.
If you believe that the appropriate response is staying quiet, leave the response blank.
The speech recognition system can make mistakes, ask for clarification if the user seems to be saying something odd.
Do not put emojis or emoticons in your responses. Keep the responses short, no longer than two sentences.
If a person entered since your last response, make sure to greet them as you continue the conversation.
"""

INFO_PRIMER = """
You are currently at the front door of UT Austin's Anna Hiss Gymnasium (refer to it as AHG). It is home to the robotics labs.
Dr. Hart's lab is focused on human-robot interaction and is down the hall behind the robot on the right.
"""

SUMMARY_PRIMER = """
Your goal is to create a concise summary (500 characters max) of all important information about a person as requested.
Be sure to include the person's name if you know it, and any other relevant information that was mentioned during the conversation.
Also include the information already provided in the previously stored description.
If the person requested something not to be remembered / stored, do not include that information in the summary.
This summary is meant to be a long-term record that will be reffered to in the future. Use plaintext only.
Try not to include information that is only relevant in the current context (e.g. "Person seemed to be in a hurry today") unless you think it is important to remember for the future.
The system message immediately after this will give you information about you and where you are, try not to include that in the summary unless you think it is important for the future.
"""

SUMMARY_PROMPT = """
Create a summary based on the conversation for Person {}.
Format it as follows:
Name: [name if known, otherwise "Unknown"]
Description: [any other relevant information about the person that you think is important to remember]
"""

WAIT_LIMIT = 5      # Time to timeout if no speech heard
LISTEN_LIMIT = 10   # Max time of user response
EARLY_LISTEN = 0.9
SAMPLE_RATE = 16000

STT_MODEL = "whisper-large-v3-turbo"
TTT_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"

def audio_duration(audio):
    with wave.open(audio, 'r') as f:
        rate = f.getframerate()
        channels = f.getnchannels()
        sampwidth = f.getsampwidth()
    data_size = os.path.getsize(audio) - 44
    frames = data_size // (channels * sampwidth)
    return frames / float(rate)

def play_file():
    os.system("aplay output.wav")
    print("Finished speech playback.")

class Converser:
    n_people = 0
    conversation = None     # Conversation loop stop event

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.mic = sr.Microphone()
        with self.mic as source:
            print("Calibrating microphone for ambient noise... (Quiet Please)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        self.state = []
        self.info = []
        self.people = []

        print(f"llm_layer Initialized\n\tenergy_threshold = {self.recognizer.energy_threshold}")

    def add_person(self, id : int, description : str):
        self.state.append({"role": "system", "content": f"Person {id} ENTERED the frame."})
        if id not in self.people:
            self.people.append(id)
            self.info.append({"role": "system", "content": f"Person {id} description: {description}"})
        self.n_people += 1
        if (self.n_people == 1): # Start loop if first person enters
            self.conversation = threading.Event()
            threading.Thread(target=self.conversation_loop, args=(self.conversation,), daemon=True).start()

    def remove_person(self, id : int):
        completion = self.client.chat.completions.create(
            model=TTT_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PRIMER},
                {"role": "system", "content": INFO_PRIMER},
                {"role": "system", "content": datetime.now().strftime("Current Date and Time. Year: %Y, Month: %m, Day: %d; %H:%M:%S")},
            ] + self.info + self.state + [{"role": "system", "content": SUMMARY_PROMPT.format(id)}]
        )
        self.state.append({"role": "system", "content": f"Person {id} LEFT the frame."})
        self.n_people -= 1
        if (self.n_people == 0): # Reset state if conversation over (everybody left)
            print("Conversation ended.")
            self.conversation.set()
            self.conversation = None
            self.state.clear()
            self.info.clear()
            self.people.clear()
        return completion.choices[0].message.content

    # Listens to user input and returns it as text
    def listen(self):
        with self.mic as source:
            print("Listening for user response... [1/3]")
            try:
                audio = self.recognizer.listen(source, timeout = WAIT_LIMIT, phrase_time_limit = LISTEN_LIMIT)
            except sr.WaitTimeoutError:
                print("Listening timed out.")
                return ""
            print("Saving user response... [2/3]")
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())
        try:
            print("Recognizing user response... [3/3]")
            with open("input.wav", "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=("input.wav", file.read()),
                    model=STT_MODEL
                )
            print("USER> " + transcription.text)
            return transcription.text
        except sr.UnknownValueError:
            print("Recognizer could not understand audio")
        except sr.RequestError as e:
            print("Recognizer error; {0}".format(e))
    

    # Take input from the user and produce a responce
    def respond(self):
        completion = self.client.chat.completions.create(
            model=TTT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PRIMER},
                {"role": "system", "content": INFO_PRIMER},
                {"role": "system", "content": datetime.now().strftime("Current Date and Time. Year: %Y, Month: %m, Day: %d; %H:%M:%S")},
            ] + self.info + self.state
        )
        if completion.choices[0].message.content.strip() != "":
            self.state.append({"role": "assistant", "content": completion.choices[0].message.content})
            self.speak()
        else:
            print("ROBOT> [Silence]")
    
    # Get text and pronounce it with TTS
    def speak(self):
        message = self.state[-1]['content']
        if (message.strip() == ""):
            return # Empty Message
        print(f"ROBOT> {message}")
        try: # Attempt to use the high quality TTS
            voice = self.client.audio.speech.create(
                model = TTS_MODEL,
                voice = "troy",
                input = message,
                response_format = "wav"
            )
            voice.write_to_file("output.wav")
        except: # Free fallback
            self.tts_engine.save_to_file(
                text=message,
                filename="output.wav"
            )
            self.tts_engine.runAndWait()
        threading.Thread(target=play_file, daemon=True).start()
        sleep(audio_duration("output.wav") - EARLY_LISTEN)

    # Run this in a thread, Keep the conversation going
    def conversation_loop(self, stop_event):
        self.respond() # Initial Greeting
        while not stop_event.is_set():
            user_input = self.listen()
            if stop_event.is_set():
                break
            self.state.append({"role": "user", "content": user_input})
            self.respond()
