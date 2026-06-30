import pyttsx3
import time

engine = pyttsx3.init()

# Make voice loud and clear
engine.setProperty('rate', 160)   # slower = clearer
engine.setProperty('volume', 1.0) # max volume

# Optional: choose better voice (Windows)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # try voices[1] if needed

last_spoken_time = 0

def speak_alert(direction, distance):
    global last_spoken_time

    # Avoid continuous spam (2 sec gap)
    if time.time() - last_spoken_time < 2:
        return

    text = f"Look at your {direction}. Someone is there at {distance} meters"

    engine.say(text)
    engine.runAndWait()

    last_spoken_time = time.time()