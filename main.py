import asyncio
import websockets
import json
import sys
import os
import requests

try:
    import openai
except ModuleNotFoundError:
    print("Error: The 'openai' module is missing. Please install it using 'pip install openai'.", file=sys.stderr)
    # You might also need to set your OpenAI API key, often via an environment variable:
    # export OPENAI_API_KEY='your-api-key'
    # Or in the script (less secure): openai.api_key = 'your-api-key'
    sys.exit(1)


GENESYS_WS_URL = "wss://EbayHelperTool.sourabh2386.repl.co/ws"
# GENESYS_API_KEY = os.getenv("GENESYS_API_KEY") # This variable seems unused?

GENESYS_CLIENT_ID = "fbf114de-7557-46ae-9080-52a9d27aaf5d"
GENESYS_CLIENT_SECRET = "clgyBjzwC5d_c30fG9wseLyKVK2iL_jVAvVdz4lLEkU" # Ensure these are correct and valid
GENESYS_AUTH_URL = "https://login.usw2.pure.cloud/oauth/token"

def get_genesys_token():
    """Authenticate with Genesys and get an access token."""
    print("Attempting to get Genesys token...")
    try:
        response = requests.post(
            GENESYS_AUTH_URL,
            data={"grant_type": "client_credentials"},
            auth=(GENESYS_CLIENT_ID, GENESYS_CLIENT_SECRET)
        )

        print(f"Genesys Auth Status Code: {response.status_code}")
        print(f"Genesys Auth Response Headers: {response.headers}")
        print(f"Genesys Auth Response Text: {response.text}") # Print the raw text response

        # Check if the request was successful (status code 2xx)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        print(f"Genesys Auth Response JSON: {response_data}") # Print the parsed JSON

        if "access_token" in response_data:
            print("Access token received successfully.")
            return response_data["access_token"]
        else:
            # If 'access_token' is not found, print an error and exit.
            print("Error: 'access_token' key not found in the Genesys response.", file=sys.stderr)
            # Check common error fields from OAuth responses
            error_details = response_data.get('error', 'Unknown error')
            error_desc = response_data.get('error_description', 'No description')
            print(f"Genesys Error: {error_details} - {error_desc}", file=sys.stderr)
            sys.exit(1) # Exit the script because we can't proceed without a token

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred during Genesys authentication: {http_err}", file=sys.stderr)
        print(f"Response content: {response.text}", file=sys.stderr) # Show response body on HTTP error
        sys.exit(1)
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred during Genesys authentication: {req_err}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from Genesys. Response text was: {response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in get_genesys_token: {e}", file=sys.stderr)
        sys.exit(1)

# --- Rest of your code ---

async def process_audio_stream():
    token = get_genesys_token()
    if not token: # Should not happen if sys.exit(1) is used above, but good practice
        print("Failed to get Genesys token. Exiting.", file=sys.stderr)
        return

    headers = {"Authorization": f"Bearer {token}"}
    print(f"Connecting to WebSocket: {GENESYS_WS_URL}")

    try:
        async with websockets.connect(GENESYS_WS_URL, extra_headers=headers) as websocket:
            print("WebSocket connection established.")
            while True:
                message = await websocket.recv()
                print("Received message from WebSocket.")
                try:
                    data = json.loads(message)
                    if "audio" not in data:
                        print("Warning: Received message without 'audio' key.", file=sys.stderr)
                        continue # Skip if no audio data

                    audio_chunk = data["audio"] # Extract audio data (assuming it's base64 encoded string or similar bytes format expected by OpenAI)

                    # Note: OpenAI transcription typically expects raw audio bytes or a file-like object.
                    # Ensure 'audio_chunk' is in the correct format. Base64 decoding might be needed.
                    # For now, assuming it's directly usable.
                    transcribed_text = transcribe_audio(audio_chunk)
                    if transcribed_text: # Only proceed if transcription was successful
                        print(f"Transcribed: {transcribed_text}")

                        response_text = generate_response(transcribed_text)
                        if response_text: # Only proceed if generation was successful
                            print(f"GPT Response: {response_text}")

                            synthesized_audio = synthesize_speech(response_text)
                            if synthesized_audio: # Only proceed if synthesis was successful
                                # Send back to Genesys (ensure format is correct, e.g., base64 encode if needed)
                                await websocket.send(json.dumps({"audio": synthesized_audio.decode('latin-1')})) # Example if bytes need string representation
                                print("Response sent to Genesys")

                except json.JSONDecodeError:
                    print(f"Error decoding JSON from WebSocket message: {message}", file=sys.stderr)
                except KeyError as ke:
                    print(f"KeyError processing WebSocket message: {ke}. Message: {message}", file=sys.stderr)
                except Exception as e:
                     print(f"Error processing WebSocket message: {e}", file=sys.stderr)


    except websockets.exceptions.ConnectionClosedOK:
        print("WebSocket connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"WebSocket connection error: {e}", file=sys.stderr)


def transcribe_audio(audio_chunk):
    """Transcribes audio chunk using Whisper."""
    # NOTE: OpenAI uses 'whisper-1' model for transcription. 'gpt-4o-transcribe' is not a valid model name.
    # NOTE: The input format needs to be correct. Often this is raw bytes written to a temporary file-like object.
    #       If audio_chunk is base64 encoded, you need to decode it first.
    # Example assuming audio_chunk is bytes:
    from io import BytesIO
    audio_file = BytesIO(audio_chunk)
    audio_file.name = "audio.wav" # OpenAI needs a filename hint for format
    try:
        # Ensure you have set your OpenAI API key
        # Make sure openai library is version < 1.0 or adapt to new syntax if >= 1.0
        # Assuming older syntax:
        # response = openai.Audio.transcribe("whisper-1", audio_file)
        # Assuming newer syntax (v1.0+):
        client = openai.OpenAI() # Assumes OPENAI_API_KEY env var is set
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        # return response["text"] # Old syntax
        return response.text # New syntax
    except Exception as e:
        print(f"Error in transcription: {e}", file=sys.stderr)
        return ""

def generate_response(transcribed_text):
    """Generates a response using gpt-4o."""
    try:
        # Ensure you have set your OpenAI API key
        # Assuming older syntax:
        # response = openai.ChatCompletion.create(...)
        # Assuming newer syntax (v1.0+):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": transcribed_text}
            ]
        )
        # return response["choices"][0]["message"]["content"] # Old syntax
        return response.choices[0].message.content # New syntax
    except Exception as e:
        print(f"Error in response generation: {e}", file=sys.stderr)
        return ""

def synthesize_speech(text_response):
    """Converts text to speech using OpenAI TTS."""
    # NOTE: Valid models are 'tts-1' and 'tts-1-hd'. 'gpt-4o-mini-tts' is not a valid model name.
    try:
         # Ensure you have set your OpenAI API key
        # Assuming older syntax:
        # response = openai.Audio.synthesize(...) - This method doesn't exist in the old library
        # Using newer syntax (v1.0+):
        client = openai.OpenAI()
        response = client.audio.speech.create(
            model="tts-1", # or "tts-1-hd"
            voice="alloy", # Example voice, choose one: alloy, echo, fable, onyx, nova, shimmer
            input=text_response
            # You might need to specify response_format, e.g., 'mp3', 'opus', 'aac', 'flac'
        )
        # response.content provides the raw audio bytes
        return response.content
    except Exception as e:
        print(f"Error in speech synthesis: {e}", file=sys.stderr)
        return b""

if __name__ == "__main__":
    # Ensure OpenAI API Key is set (e.g., via environment variable OPENAI_API_KEY)
    if not os.getenv("OPENAI_API_KEY"):
         print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
         # sys.exit(1) # Optional: exit if key is missing

    asyncio.run(process_audio_stream())
