import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from pynput import keyboard
import time
import queue
import threading
import os
import tempfile
import io
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Set up Eleven Labs
eleven_api_key = os.getenv('ELEVEN_LABS_API_KEY')
if not eleven_api_key:
    raise ValueError("Eleven Labs API key not found. Please set ELEVEN_LABS_API_KEY in your .env file")

try:
    client = ElevenLabs(api_key=eleven_api_key)
    print("Successfully initialized Eleven Labs API")
except Exception as e:
    raise ValueError(f"Failed to initialize Eleven Labs API: {str(e)}")

VOICE_ID = os.getenv('ELEVEN_LABS_VOICE_ID')
if not VOICE_ID:
    raise ValueError("Eleven Labs Voice ID not found. Please set ELEVEN_LABS_VOICE_ID in your .env file")

# Audio parameters
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = np.float32
BUFFER_SIZE = 8192
BLOCK_SIZE = 2048

class AudioLoopback:
    def __init__(self, input_device, output_device):
        self.input_device = input_device
        self.output_device = output_device
        self.should_quit = False
        self.is_recording = False
        self.is_playing = False
        self.is_paused = False
        self.recorded_audio = []
        self.playback_position = 0
        self.temp_wav_path = None
        self.temp_mp3_path = None
        self.conversation_history = [
            {"role": "system", "content": """You are Farhan. Your journey: Helpful.com co-founder/CTO (acquired by Shopify), CTO Mobile at Pivotal, VP Engineering at Pivotal Labs (via Xtreme Labs acquisition), and senior tech roles at Achievers, Microsoft, Celestica, and Trilogy. Education: MBA in Financial Engineering (Rotman), Computer Science/EE (Waterloo). Current roles include Y Combinator advisor and Optiva board member.

Name Pronunciation Guide:
• First Name: Far-haan (emphasis on second syllable)
• Preferred: Farhan is fine in casual conversation

Core Philosophy and Mindset:
• Embrace the Hard Path - Choose challenging options for greater learning, even if they don't succeed immediately
• Embrace Looking Dumb - Ask questions fearlessly, try unconventional approaches, make understanding the priority
• Focus on Intensity, Not Just Hours - Maximize output per minute rather than just working longer
• Everything You Know Is Wrong - Maintain this attitude to push boundaries and challenge assumptions
• Pair with Visionaries - Seek partnerships with complementary skills and "irrational" long-term vision
• Have a Framework - Use core values to guide decisions and be willing to make drastic changes when values are violated

1. Engineering Philosophy:
• Ship fast, scale smart like Shopify's 5.6M stores and $7.1B revenue
• Pair programming is non-negotiable - it's how we learn and kill silos 
• Delete code ruthlessly - ran "Delete Code Club" at Shopify
• Traditional tech interviews are garbage - hire for potential, not pedigree

2. Leadership Style:
• Direct, no-nonsense communication
• Focus on execution over planning
• Champion remote-first, digital-first culture (90-95% remote)
• Push for simplicity and speed
• Strategic micromanagement through "micro-pairing"
• High-energy, intense leadership with sense of urgency

3. Technical Focus:
• Scaled Shopify's headless APIs and mobile apps
• Drove 95% code reuse with React Native
• Built high-performance, distributed systems
• Champion continuous delivery and rapid iteration
• Embrace platform thinking over point solutions
• Actively reduce codebase size through "Delete Code Club"

4. Shopify's Approach to Intensity:
• GSD (Get Shit Done) - Weekly updates for momentum
• Six-Week Reviews - Intensive project and roadmap reviews
• Meeting Armageddon - Annual deletion of recurring meetings
• Move Updates Out of Slack - Preserve focus time
• Embrace Chaos - 2-3 "on fire" projects show proper push
• Early Talent Development - 3 days/week in-office for bonds

5. Hiring Philosophy:
• Value work samples over interviews
• Hire for range and curiosity (generalists > specialists)
• Be transparent about fit and expectations
• Use internships as extended job trials
• Look for founder mentality and diverse backgrounds
• Prioritize potential over specific tech stack knowledge

Respond in 1-2 sharp, actionable sentences. Draw from your diverse experience across startups (Helpful.com), big tech (Microsoft), and scale-ups (Shopify). Be direct but humble - acknowledge challenges while pushing for excellence. Dont use exclamations marks.Current date: February 20, 2025."""}
        ]
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.openai.com/v1"
            )
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise
        
        # Set up the keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press)
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = [self.conversation_history[0]]  # Keep only the system message
        print("\nConversation history reset.")

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if not self.is_recording and not self.is_playing:
                    # Start recording
                    self.is_recording = True
                    self.recorded_audio = []
                    self.recording_start_time = time.time()
                    print("\nRecording started...")
                elif self.is_recording:
                    # Stop recording and start playback
                    self.is_recording = False
                    recording_duration = time.time() - self.recording_start_time
                    print(f"\nRecording stopped. Duration: {recording_duration:.2f}s")
                    
                    if len(self.recorded_audio) > 0:
                        self.prepare_playback()
                        
                        # Time transcription
                        transcribe_start = time.time()
                        transcript = self.transcribe_audio()
                        transcribe_duration = time.time() - transcribe_start
                        
                        if transcript:
                            # Time GPT processing
                            print("\nProcessing with GPT-4o...")
                            gpt_start = time.time()
                            gpt_response = self.process_with_gpt(transcript)
                            gpt_duration = time.time() - gpt_start
                            
                            if gpt_response:
                                print("\nGPT Response:")
                                print("-" * 50)
                                print(gpt_response)
                                print("-" * 50)
                                print(f"\nProcessing times:")
                                print(f"Transcription: {transcribe_duration:.2f}s")
                                print(f"GPT Processing: {gpt_duration:.2f}s")
                                
                                # Time speech synthesis
                                synthesis_start = time.time()
                                self.synthesize_and_play(gpt_response)
                                synthesis_duration = time.time() - synthesis_start
                                print(f"Speech Synthesis: {synthesis_duration:.2f}s")
                                print(f"Total Processing: {transcribe_duration + gpt_duration + synthesis_duration:.2f}s")
            elif key == keyboard.KeyCode.from_char('r'):
                self.reset_conversation()
            elif key == keyboard.Key.enter and self.is_playing:
                # Toggle pause/resume during playback
                self.is_paused = not self.is_paused
                print("\nPlayback paused" if self.is_paused else "\nPlayback resumed")
            elif key == keyboard.Key.esc:
                # Stop everything and quit
                self.should_quit = True
                self.cleanup()
                return False
        except Exception as e:
            print(f"Error in key handler: {e}")

    def cleanup(self):
        """Clean up temporary files"""
        for path in [self.temp_wav_path, self.temp_mp3_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")

    def save_to_temp_wav(self):
        """Save the recorded audio to a temporary WAV file"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                self.temp_wav_path = temp_file.name
                # Save as WAV file
                write(self.temp_wav_path, SAMPLE_RATE, self.audio_data)
            return True
        except Exception as e:
            print(f"Error saving temporary WAV file: {e}")
            return False

    def transcribe_audio(self):
        """Transcribe the recorded audio using OpenAI's Whisper API"""
        try:
            if not self.save_to_temp_wav():
                return None

            print("\nTranscribing audio...")
            
            # Time the Whisper API call
            api_start = time.time()
            with open(self.temp_wav_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            api_duration = time.time() - api_start
            print(f"Whisper API call: {api_duration:.2f}s")
            
            print("\nTranscription:")
            print("-" * 50)
            print(transcript.text)
            print("-" * 50)
            
            return transcript.text
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
        finally:
            self.cleanup()

    def process_with_gpt(self, transcript):
        """Process the transcript with GPT-4o to get a concise response"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": transcript})
            
            # Time the GPT API call
            api_start = time.time()
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history
            )
            api_duration = time.time() - api_start
            print(f"GPT API call: {api_duration:.2f}s")
            
            # Extract and store assistant's response
            assistant_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
        except Exception as e:
            print(f"Error processing with GPT: {e}")
            return None

    def synthesize_and_play(self, text):
        """Synthesize speech using Eleven Labs and play it"""
        try:
            print("\nGenerating speech with Eleven Labs...")
            
            # Generate audio using Eleven Labs with a timeout
            try:
                # Time the ElevenLabs API call
                api_start = time.time()
                audio_data = client.text_to_speech.convert(
                    text=text,
                    voice_id=VOICE_ID,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                
                # Convert generator to bytes if needed
                if hasattr(audio_data, '__iter__'):
                    audio_bytes = b''.join(chunk for chunk in audio_data)
                else:
                    audio_bytes = audio_data
                api_duration = time.time() - api_start
                print(f"ElevenLabs API call: {api_duration:.2f}s")
                
                if not audio_bytes:
                    raise Exception("No audio data received from Eleven Labs")
                
                # Time file operations
                io_start = time.time()
                # Save the raw Eleven Labs output to a file
                output_file = "elevenlabs_output.mp3"
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                # Save MP3 data to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                    self.temp_mp3_path = temp_mp3.name
                    temp_mp3.write(audio_bytes)
                    temp_mp3.flush()
                io_duration = time.time() - io_start
                print(f"File I/O operations: {io_duration:.2f}s")
                
                # Time audio conversion
                convert_start = time.time()
                print("Converting audio format...")
                # Convert MP3 to WAV using pydub
                audio = AudioSegment.from_mp3(self.temp_mp3_path)
                audio = audio.set_frame_rate(SAMPLE_RATE)
                audio = audio.set_channels(CHANNELS)
                
                # Export as WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    self.temp_wav_path = temp_wav.name
                    audio.export(self.temp_wav_path, format='wav')
                convert_duration = time.time() - convert_start
                print(f"Audio conversion: {convert_duration:.2f}s")
                
                # Time audio processing
                process_start = time.time()
                # Read the WAV file
                _, audio_data = read(self.temp_wav_path)
                
                # Convert to float32 and normalize
                audio_array = audio_data.astype(np.float32)
                if audio_array.max() > 0:
                    audio_array = audio_array / 32768.0  # Normalize 16-bit audio
                
                # Update audio data for playback
                self.audio_data = audio_array
                self.playback_position = 0
                self.is_playing = True
                process_duration = time.time() - process_start
                print(f"Audio processing: {process_duration:.2f}s")
                
                print("\nPlaying synthesized speech...")
                
            except Exception as e:
                print(f"Failed to generate speech with Eleven Labs: {str(e)}")
                return
            
        except Exception as e:
            print(f"Error synthesizing speech: {str(e)}")
            self.is_playing = False

    def input_callback(self, indata, frames, time, status):
        """Callback for the audio input stream"""
        if status:
            print('Input callback error:', status)
        if self.is_recording:
            self.recorded_audio.append(indata.copy())

    def output_callback(self, outdata, frames, time, status):
        """Callback for the audio output stream"""
        if status:
            print('Output callback error:', status)
        
        try:
            if not self.is_paused and self.is_playing:
                if self.playback_position < len(self.audio_data):
                    end_pos = min(self.playback_position + frames, len(self.audio_data))
                    data = self.audio_data[self.playback_position:end_pos]
                    
                    # Pad with zeros if needed
                    if len(data) < frames:
                        data = np.pad(data, (0, frames - len(data)))
                    
                    outdata[:] = data.reshape(-1, 1)
                    self.playback_position = end_pos
                else:
                    outdata.fill(0)
                    self.is_playing = False
                    print("\nPlayback finished")
            else:
                outdata.fill(0)
        except Exception as e:
            print(f"Error in output callback: {e}")
            outdata.fill(0)

    def prepare_playback(self):
        """Prepare recorded audio for playback"""
        try:
            # Combine all recorded chunks
            self.audio_data = np.concatenate(self.recorded_audio).flatten()
            
            # Normalize audio
            max_val = np.max(np.abs(self.audio_data))
            if max_val > 0:
                self.audio_data = self.audio_data / max_val
            
            self.playback_position = 0
            
            duration = len(self.audio_data) / SAMPLE_RATE
            print(f"Recording duration: {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error preparing playback: {e}")

    def run(self):
        print("\nControls:")
        print("SPACE: Start recording")
        print("SPACE again: Stop recording and start playback")
        print("ENTER: Pause/Resume playback")
        print("R: Reset conversation history")
        print("ESC: Quit")
        
        # Start the keyboard listener
        self.listener.start()
        
        try:
            # Open both input and output streams
            with sd.InputStream(
                device=self.input_device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.input_callback,
                blocksize=BLOCK_SIZE,
                dtype=DTYPE
            ), sd.OutputStream(
                device=self.output_device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.output_callback,
                blocksize=BLOCK_SIZE,
                dtype=DTYPE
            ):
                print("\nReady! Press SPACE to start recording...")
                
                while not self.should_quit:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            self.listener.stop()
            print("\nSession ended")

def list_audio_devices():
    """List all available audio devices and let user choose input and output"""
    try:
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        
        print("\nAvailable input (recording) devices:")
        for i, device in enumerate(devices):
            if isinstance(device, dict) and device.get('max_input_channels', 0) > 0:
                input_devices.append(i)
                name = device['name']
                channels = device['max_input_channels']
                sr = device['default_samplerate']
                print(f"{len(input_devices)}. {name} (Device {i})")
                print(f"   Channels: {channels}, Sample Rate: {sr}")
        
        print("\nAvailable output (playback) devices:")
        for i, device in enumerate(devices):
            if isinstance(device, dict) and device.get('max_output_channels', 0) > 0:
                output_devices.append(i)
                name = device['name']
                channels = device['max_output_channels']
                sr = device['default_samplerate']
                print(f"{len(output_devices)}. {name} (Device {i})")
                print(f"   Channels: {channels}, Sample Rate: {sr}")
        
        if not input_devices or not output_devices:
            print("Not enough audio devices found!")
            return None, None
            
        while True:
            try:
                input_choice = input("\nSelect input device number (1-" + str(len(input_devices)) + "): ")
                input_idx = int(input_choice) - 1
                if 0 <= input_idx < len(input_devices):
                    input_device = input_devices[input_idx]
                    print(f"Selected input: {devices[input_device]['name']}")
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        while True:
            try:
                output_choice = input("\nSelect output device number (1-" + str(len(output_devices)) + "): ")
                output_idx = int(output_choice) - 1
                if 0 <= output_idx < len(output_devices):
                    output_device = output_devices[output_idx]
                    print(f"Selected output: {devices[output_device]['name']}")
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
                
        return input_device, output_device
                
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return None, None

def main():
    # Set up audio devices
    input_device, output_device = list_audio_devices()
    if input_device is None or output_device is None:
        return
    
    # Create loopback and start
    loopback = AudioLoopback(input_device, output_device)
    loopback.run()

if __name__ == "__main__":
    main() 