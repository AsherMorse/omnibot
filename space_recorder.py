import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pynput import keyboard
import time
from datetime import datetime
import queue
import threading

# Audio parameters
SAMPLE_RATE = 48000  # Higher sample rate for better quality
CHANNELS = 1  # Mono recording
DTYPE = np.int16  # Better compatibility with WAV format
CHUNK_SIZE = 1024  # Larger chunk size for better performance
BUFFER_SIZE = 100000  # Large buffer to prevent data loss

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=BUFFER_SIZE)
        self.is_recording = False
        self.should_quit = False
        
        # Set up the keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press)
    
    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if not self.is_recording:
                    # Start recording
                    self.is_recording = True
                    self.audio_queue.queue.clear()  # Clear previous recording
                    print("\nRecording started...")
                else:
                    # Stop recording
                    self.is_recording = False
                    print("\nRecording stopped.")
                    self.save_recording()
            elif key == keyboard.Key.esc:
                self.should_quit = True
                if self.is_recording:
                    self.is_recording = False
                    self.save_recording()
                return False  # Stop listener
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def audio_callback(self, indata, frames, time, status):
        """This is called for each audio block"""
        if status:
            print('Audio callback error:', status)
        if self.is_recording:
            try:
                self.audio_queue.put(indata.copy())
            except queue.Full:
                print('Buffer overflow - some audio data was lost!')
    
    def save_recording(self):
        try:
            if self.audio_queue.empty():
                print("No audio recorded!")
                return

            print("\nProcessing recording...")
            
            # Collect all audio data from the queue
            audio_data = []
            while not self.audio_queue.empty():
                audio_data.append(self.audio_queue.get())
            
            # Combine all chunks
            final_recording = np.concatenate(audio_data)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
            # Save as WAV file
            write(filename, SAMPLE_RATE, final_recording)
            print(f"Recording saved as {filename}")
            
            duration = len(final_recording) / SAMPLE_RATE
            print(f"Recording duration: {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def record_audio(self):
        print("\nPress SPACE once to start recording.")
        print("Press SPACE again to stop recording.")
        print("Press ESC to quit.")
        
        # Start the keyboard listener
        self.listener.start()
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self.audio_callback,
                blocksize=CHUNK_SIZE
            ):
                print("\nAudio stream started. Ready to record.")
                
                while not self.should_quit:
                    time.sleep(0.1)  # Reduced CPU usage while waiting
                    
        except KeyboardInterrupt:
            print("\nRecording interrupted.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            self.listener.stop()
            print("\nRecording session ended.")

def list_audio_devices():
    """List all available audio input devices and let user choose one"""
    try:
        devices = sd.query_devices()
        input_devices = []
        
        print("\nAvailable input devices:")
        for i, device in enumerate(devices):
            if isinstance(device, dict) and device.get('max_input_channels', 0) > 0:
                input_devices.append(i)
                name = device['name']
                channels = device['max_input_channels']
                sr = device['default_samplerate']
                print(f"{len(input_devices)}. {name} (Device {i})")
                print(f"   Channels: {channels}, Sample Rate: {sr}")
        
        if not input_devices:
            print("No input devices found!")
            return None
            
        while True:
            try:
                choice = input("\nSelect input device number (1-" + str(len(input_devices)) + "): ")
                idx = int(choice) - 1
                if 0 <= idx < len(input_devices):
                    device_id = input_devices[idx]
                    device = devices[device_id]
                    print(f"\nSelected: {device['name']}")
                    
                    # Adjust sample rate if needed
                    global SAMPLE_RATE
                    if device['default_samplerate'] != SAMPLE_RATE:
                        SAMPLE_RATE = int(device['default_samplerate'])
                        print(f"Adjusted sample rate to: {SAMPLE_RATE} Hz")
                    
                    return device_id
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
                
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return None

def setup_audio_device():
    try:
        device_id = list_audio_devices()
        if device_id is not None:
            sd.default.device = device_id
            return True
        return False
        
    except Exception as e:
        print(f"Error setting up audio device: {e}")
        return False

if __name__ == "__main__":
    if setup_audio_device():
        recorder = AudioRecorder()
        recorder.record_audio()