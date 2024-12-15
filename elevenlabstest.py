import os
from elevenlabs import ElevenLabs, play


from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
# Initialize the ElevenLabs client with your API key
client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

# Clone the voice using your sample recording
voice1 = client.clone(
    name="ClonedVoice",
    description="A cloned voice from the provided sample.",
    files=["samplespeech.mp3"],  # Replace with the path to your voice sample
      # Optional: removes background noise if present
)

# Generate speech using the cloned voice
audio1 = client.generate(
    text="Hello! This is a test of Dad's cloned voice.",
    voice=voice1
)

voice2 = client.clone(
    name="ClonedVoice",
    description="A cloned voice from the provided sample.",
    files=["dadsamplespeech.mp3"],  # Replace with the path to your voice sample
      # Optional: removes background noise if present
)

# Generate speech using the cloned voice
audio2 = client.generate(
    text="Hello! This is a test of Kevin's cloned voice.",
    voice=voice2
)

voice3 = client.clone(
    name="ClonedVoice",
    description="A cloned voice from the provided sample.",
    files=["dadsamplespeech.mp3"],  # Replace with the path to your voice sample
      # Optional: removes background noise if present
)

# Generate speech using the cloned voice
audio3 = client.generate(
    text="Hello! This is a test of Mom's cloned voice.",
    voice=voice3
)

# Play the generated audio
play(audio1)
play(audio2)
play(audio3)
