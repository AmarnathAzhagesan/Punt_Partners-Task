# main.py
import argparse
import pyaudio
from sdk import VoiceBotSDK

def main():
    parser = argparse.ArgumentParser(description='Voice Bot SDK CLI')
    parser.add_argument('--stt-key', required=True, help='Deepgram API Key for STT')
    parser.add_argument('--tts-key', required=True, help='Deepgram or OpenAI API Key for TTS')
    parser.add_argument('--llm-key', required=True, help='OpenAI API Key for LLM')
    args = parser.parse_args()

    # Initialize SDK
    sdk = VoiceBotSDK()
    sdk.setup(
        stt_config={'name': 'deepgram', 'api_key': args.stt_key},
        tts_config={'name': 'openai', 'api_key': args.tts_key},
        llm_config={'name': 'openai', 'api_key': args.llm_key, 'system_prompt': 'You are a helpful assistant.'}
    )

    # Configure PyAudio streams
    p = pyaudio.PyAudio()
    input_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    output_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True, frames_per_buffer=1024)

    # Start conversation
    sdk.stream_conversation(input_stream, output_stream)

if __name__ == '__main__':
    main()
