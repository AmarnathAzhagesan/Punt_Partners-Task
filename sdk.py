# sdk.py
import openai
from deepgram import Deepgram
import time

class VoiceBotSDK:
    def __init__(self):
        self.stt_engine = None
        self.tts_engine = None
        self.llm_engine = None
        self.system_prompt = ''

    def setup(self, stt_config, tts_config, llm_config):
        # Initialize STT engine
        if stt_config['name'].lower() == 'deepgram':
            self.stt_engine = Deepgram(stt_config['api_key'])

        # Initialize TTS engine
        if tts_config['name'].lower() in ['deepgram', 'openai']:
            self.tts_engine = {
                'name': tts_config['name'],
                'api_key': tts_config['api_key']
            }

        # Initialize LLM engine
        if llm_config['name'].lower() == 'openai':
            openai.api_key = llm_config['api_key']
            self.llm_engine = openai

        # Store system prompt
        self.system_prompt = llm_config.get('system_prompt', '')

    def stream_conversation(self, input_stream, output_stream):
        # Read input from the input stream and convert to text
        start_time = time.time()
        text = self.stt_engine.transcribe_stream(input_stream)
        stt_end_time = time.time()

        # Query LLM with the converted text
        response = self.llm_engine.Completion.create(
            engine="text-davinci-003",
            prompt=f"{self.system_prompt}\n\nUser: {text}\nBot:",
            max_tokens=150
        )
        llm_response_time = time.time()

        # Convert LLM response text to speech
        if self.tts_engine['name'].lower() == 'deepgram':
            tts_output = self.stt_engine.speak(response['choices'][0]['text'])
        else:
            tts_output = openai.Audio.create(
                text=response['choices'][0]['text'],
                model="davinci"
            )
        tts_end_time = time.time()

        # Stream the TTS output to the output stream
        output_stream.write(tts_output)

        # Print performance metrics
        print(f"STT Time: {stt_end_time - start_time:.2f} seconds")
        print(f"LLM Response Time: {llm_response_time - stt_end_time:.2f} seconds")
        print(f"TTS Time: {tts_end_time - llm_response_time:.2f} seconds")
