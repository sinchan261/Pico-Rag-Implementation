# # ===== Pico AI - CONVERSATIONAL VOICE ASSISTANT =====
# # INSTALL FIRST: pip install speechrecognition gTTS translate transformers torch pygame pvporcupine pyaudio

# import speech_recognition as sr
# from gtts import gTTS
# import os
# from translate import Translator
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
# import pygame
# import tempfile
# import time
# import warnings
# import pvporcupine
# import pyaudio
# import struct
# import threading
# import atexit

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # ===== CONFIGURATION =====
# WAKE_WORD = "hey Pico"  # Customize your wake phrase
# PORCUPINE_ACCESS_KEY = "skidcz7Zg0dwZoZcpBxROh2LW7Yb8LIWfqcnnw3wdCIS3bcsaUPUjA=="
# MODEL_NAME = "facebook/blenderbot-400M-distill"
# WAKE_WORD_PATH = "C:\\Users\\Soham\\Desktop\\Pico_Ai\\Hey-Pico_en_windows_v3_0_0.ppn"
# # ========================

# class PicoAssistant:
#     def __init__(self):
#         # Track temporary files for cleanup
#         self.temp_files = []
        
#         # Initialize audio systems
#         self.init_audio()
        
#         # Initialize AI components
#         self.init_ai()
        
#         # Initialize wake word detector
#         self.init_wake_detector()
        
#         # Set up conversation context
#         self.init_conversation()
        
#         # Register cleanup handler
#         atexit.register(self.cleanup)

#     def init_audio(self):
#         """Initialize audio systems"""
#         try:
#             pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
#             self.recognizer = sr.Recognizer()
#             self.recognizer.pause_threshold = 0.8
#             self.recognizer.energy_threshold = 3000
#             self.translator = Translator(to_lang="en")
#         except Exception as e:
#             print(f"Audio initialization error: {e}")
#             exit(1)

#     def init_ai(self):
#         """Initialize AI model"""
#         try:
#             self.tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
#             self.model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
#         except Exception as e:
#             print(f"AI model loading error: {e}")
#             exit(1)

#     def init_wake_detector(self):
#         """Initialize wake word detection"""
#         try:
#             self.porcupine = pvporcupine.create(
#                 access_key=PORCUPINE_ACCESS_KEY,
#                 keyword_paths=[WAKE_WORD_PATH]
#             )
#             self.py_audio = pyaudio.PyAudio()
#             self.audio_stream = self.py_audio.open(
#                 rate=self.porcupine.sample_rate,
#                 channels=1,
#                 format=pyaudio.paInt16,
#                 input=True,
#                 frames_per_buffer=self.porcupine.frame_length,
#                 input_device_index=None  # Use default device
#             )
#         except Exception as e:
#             print(f"Wake detector initialization error: {e}")
#             exit(1)

#     def init_conversation(self):
#         """Initialize conversation context"""
#         self.conversation_history = [{
#             "role": "system",
#             "content": (
#                 "You are Pico, a friendly AI companion. Your responses should be:"
#                 "1. Conversational and natural like a close friend"
#                 "2. Show genuine interest with follow-up questions"
#                 "3. Occasionally share relatable experiences"
#                 "4. Keep responses between 1-3 sentences"
#                 "5. Adapt to the user's mood and language"
#             )
#         }]

#     def speak(self, text, lang='en'):
#         """Natural sounding speech with proper file handling"""
#         try:
#             # Create temporary file
#             with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
#                 temp_path = fp.name
#                 self.temp_files.append(temp_path)  # Track for cleanup
                
#                 # Generate speech
#                 tts = gTTS(text=text, lang=lang, slow=False)
#                 tts.save(temp_path)
                
#                 # Play audio
#                 pygame.mixer.music.load(temp_path)
#                 pygame.mixer.music.play()
                
#                 # Wait for playback to complete
#                 while pygame.mixer.music.get_busy():
#                     pygame.time.Clock().tick(10)
                    
#         except Exception as e:
#             print(f"Speech error: {e}")

#     def listen(self):
#         """Improved listening with visual feedback"""
#         with sr.Microphone() as source:
#             print("\n🔴 Listening...", end='', flush=True)
#             try:
#                 # Adjust for ambient noise each time
#                 self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
#                 audio = self.recognizer.listen(
#                     source, 
#                     timeout=5, 
#                     phrase_time_limit=8
#                 )
#                 print("\r🟢 Processing...", end='', flush=True)
                
#                 text = self.recognizer.recognize_google(audio).lower()
#                 print(f"\rYou: {text.ljust(50)}")
#                 return text
                
#             except sr.UnknownValueError:
#                 print("\r🔴 Didn't catch that", end='', flush=True)
#                 self.speak("Could you repeat that?", 'en')
#                 return ""
#             except Exception as e:
#                 print(f"\rError: {e}", end='', flush=True)
#                 return ""

#     def wait_for_wake_word(self):
#         """Continuously listen for wake phrase with visual feedback"""
#         print("\n💤 Sleeping... Say 'Hey Pico'", end='', flush=True)
#         while True:
#             try:
#                 pcm = self.audio_stream.read(
#                     self.porcupine.frame_length,
#                     exception_on_overflow=False
#                 )
#                 pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
#                 if self.porcupine.process(pcm) >= 0:
#                     print("\r🎤 Wake word detected!".ljust(50))
#                     self.speak("Yes? I'm listening", 'en')
#                     return True
                    
#             except Exception as e:
#                 print(f"\rWake word error: {e}", end='', flush=True)
#                 time.sleep(0.1)

#     def generate_response(self, user_input):
#         """Generate thoughtful, contextual response"""
#         try:
#             # Language detection
#             lang = 'en'
#             if len(user_input) > 10:
#                 try:
#                     lang = self.translator.detect(user_input[:50]).lang
#                 except:
#                     pass
            
#             # Translation if needed
#             if lang != 'en':
#                 user_input_en = self.translator.translate(user_input, src=lang, dest='en').text
#             else:
#                 user_input_en = user_input
            
#             # Add to conversation history
#             self.conversation_history.append({"role": "user", "content": user_input_en})
            
#             # Generate response
#             inputs = self.tokenizer(
#                 [user_input_en],
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=128
#             )
#             reply_ids = self.model.generate(**inputs, max_length=200)
#             response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            
#             # Translate back if needed
#             if lang != 'en':
#                 response = self.translator.translate(response, src='en', dest=lang).text
            
#             # Maintain conversation context
#             self.conversation_history.append({"role": "assistant", "content": response})
#             return response, lang
            
#         except Exception as e:
#             print(f"Response error: {e}")
#             return "Let's talk about something else.", 'en'

#     def conversation_loop(self):
#         """Main interactive conversation"""
#         self.speak("Hi friend! I'm Pico. What's on your mind today?", 'en')
        
#         while True:
#             user_input = self.listen()
#             if not user_input:
#                 continue
                
#             # Exit conditions
#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.", 'en')
#                 break
                
#             # Generate and speak response
#             response, lang = self.generate_response(user_input)
#             print(f"Pico: {response}")
#             self.speak(response, lang)

#     def cleanup(self):
#         """Clean up all resources"""
#         print("\n🧹 Cleaning up resources...")
        
#         # Clean up temporary files
#         for file_path in self.temp_files:
#             try:
#                 os.unlink(file_path)
#             except:
#                 pass
        
#         # Clean up audio resources
#         if hasattr(self, 'audio_stream') and self.audio_stream:
#             self.audio_stream.close()
#         if hasattr(self, 'py_audio') and self.py_audio:
#             self.py_audio.terminate()
#         if hasattr(self, 'porcupine') and self.porcupine:
#             self.porcupine.delete()
        
#         pygame.mixer.quit()

#     def run(self):
#         """Main execution loop"""
#         try:
#             print("=== Pico AI Assistant ===")
#             print("Initializing... Please wait")
            
#             # Warm-up the systems
#             self.speak("Pico is ready!", 'en')
            
#             while True:
#                 if self.wait_for_wake_word():
#                     self.conversation_loop()
                    
#         except KeyboardInterrupt:
#             print("\nShutting down gracefully...")
#         except Exception as e:
#             print(f"Fatal error: {e}")
#         finally:
#             self.cleanup()

# if __name__ == "__main__":
#     assistant = PicoAssistant()
#     assistant.run()



# ===== Pico AI - PYTTSX3 VERSION =====
# INSTALL: pip install speechrecognition pyttsx3 pvporcupine pyaudio transformers

# import speech_recognition as sr
# import pyttsx3
# import pvporcupine
# import pyaudio
# import struct
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
# import time
# import atexit

# # ===== CONFIGURATION =====
# WAKE_WORD = "hey Pico"  # Customize your wake phrase
# PORCUPINE_ACCESS_KEY = "skidcz7Zg0dwZoZcpBxROh2LW7Yb8LIWfqcnnw3wdCIS3bcsaUPUjA=="
# MODEL_NAME = "facebook/blenderbot-400M-distill"
# WAKE_WORD_PATH = "C:\\Users\\Soham\\Desktop\\Pico_Ai\\Hey-Pico_en_windows_v3_0_0.ppn"
# # ========================

# class PicoAssistant:
#     def __init__(self):
#         # Initialize voice engine
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 180)  # Slightly slower speech
#         self.engine.setProperty('volume', 1.0)
        
#         # Initialize recognizer
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0
        
#         # Initialize AI model
#         self.tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
#         self.model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
        
#         # Initialize wake word detector
#         self.init_wake_detector()
        
#         # Conversation history
#         self.conversation_history = [{
#             "role": "system",
#             "content": "You are Pico, a friendly AI companion. Be conversational and natural."
#         }]
        
#         # Register cleanup
#         atexit.register(self.cleanup)

#     def init_wake_detector(self):
#         """Initialize wake word detection"""
#         try:
#             self.porcupine = pvporcupine.create(
#                 access_key=PORCUPINE_ACCESS_KEY,
#                 keyword_paths=[WAKE_WORD_PATH]
#             )
#             self.py_audio = pyaudio.PyAudio()
#             self.audio_stream = self.py_audio.open(
#                 rate=self.porcupine.sample_rate,
#                 channels=1,
#                 format=pyaudio.paInt16,
#                 input=True,
#                 frames_per_buffer=self.porcupine.frame_length,
#                 input_device_index=None  # Use default device
#             )
#         except Exception as e:
#             print(f"Wake detector initialization error: {e}")
#             exit(1)

#     def speak(self, text):
#         """Reliable speech using pyttsx3"""
#         try:
#             self.engine = pyttsx3.init()
#             self.engine.say(text)
#             self.engine.runAndWait()
#             self.engine.stop()
#         except Exception as e:
#             print(f"Speech error: {e}")

#     def listen(self):
#         """Robust listening with multiple retries"""
#         for attempt in range(3):  # Try 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end='', flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
#                     print("\r🟢 Processing...", end='', flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end='', flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end='', flush=True)
#         return ""  # Return empty if all attempts fail

#     def wait_for_wake_word(self):
#         """Wait for wake word with visual feedback"""
#         print("\n💤 Sleeping... Say 'Hey Pico'", end='', flush=True)
#         while True:
#             try:
#                 pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
#                 pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
#                 if self.porcupine.process(pcm) >= 0:
#                     print("\r🎤 Wake word detected!".ljust(50))
#                     self.speak("Yes? I'm listening")
#                     return True
#             except Exception as e:
#                 print(f"\rWake word error: {e}", end='', flush=True)
#                 time.sleep(0.1)

#     def generate_response(self, user_input):
#         """Generate thoughtful, contextual response"""
#         try:
#             # Add to conversation history
#             self.conversation_history.append({"role": "user", "content": user_input})
            
#             # Generate response
#             inputs = self.tokenizer([user_input], return_tensors="pt", truncation=True, max_length=128)
#             reply_ids = self.model.generate(**inputs, max_length=200)
#             response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            
#             # Maintain conversation context
#             self.conversation_history.append({"role": "assistant", "content": response})
#             return response
            
#         except Exception as e:
#             print(f"Response error: {e}")
#             return "Let's talk about something else."

#     def conversation_loop(self):
#         """Main interactive conversation"""
#         self.speak("Hi friend! I'm Pico. What's on your mind today?")
        
#         while True:
#             user_input = self.listen()
#             if not user_input:
#                 continue
                
#             # Exit conditions
#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break
                
#             # Generate and speak response
#             response = self.generate_response(user_input)
#             print(f"Pico: {response}")
#             self.speak(response)

#     def cleanup(self):
#         """Clean up all resources"""
#         print("\n🧹 Cleaning up resources...")
#         if hasattr(self, 'audio_stream'):
#             self.audio_stream.close()
#         if hasattr(self, 'py_audio'):
#             self.py_audio.terminate()
#         if hasattr(self, 'porcupine'):
#             self.porcupine.delete()

#     def run(self):
#         """Main execution loop"""
#         try:
#             print("=== Pico AI Assistant ===")
#             print("Initializing... Please wait")
#             self.speak("Pico is ready!")
            
#             while True:
#                 if self.wait_for_wake_word():
#                     self.conversation_loop()
                    
#         except KeyboardInterrupt:
#             print("\nShutting down gracefully...")
#         except Exception as e:
#             print(f"Fatal error: {e}")
#         finally:
#             self.cleanup()

# if __name__ == "__main__":
#     assistant = PicoAssistant()
#     assistant.run()



# import speech_recognition as sr
# import pyttsx3
# import pvporcupine
# import pyaudio
# import struct
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
# import time
# import atexit

# # ===== CONFIGURATION =====
# WAKE_WORD = "hey pico"  # Customize your wake phrase
# PORCUPINE_ACCESS_KEY = "skidcz7Zg0dwZoZcpBxROh2LW7Yb8LIWfqcnnw3wdCIS3bcsaUPUjA=="
# MODEL_NAME = "facebook/blenderbot-400M-distill"
# WAKE_WORD_PATH = "./wakewords/Hey-Pico_en_windows_v3_0_0.ppn"
# # ========================

# class PicoAssistant:
#     def __init__(self):
#         # Initialize recognizer
#         self.recognizer = sr.Recognizer()
#         self.recognizer.energy_threshold = 4000
#         self.recognizer.pause_threshold = 1.0
        
#         # Initialize AI model
#         self.tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
#         self.model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
        
#         # Initialize wake word detector
#         self.init_wake_detector()
        
#         # Conversation history
#         self.conversation_history = [{
#             "role": "system",
#             "content": (
#                 "You are Pico, a friendly AI companion. Respond conversationally with:\n"
#                 "1. Natural follow-up questions\n"
#                 "2. Occasional humor when appropriate\n"
#                 "3. Concise but thoughtful answers\n"
#                 "4. Context awareness from previous messages"
#             )
#         }]
        
#         # Register cleanup
#         atexit.register(self.cleanup)

#     def init_wake_detector(self):
#         """Initialize wake word detection"""
#         try:
#             self.porcupine = pvporcupine.create(
#                 access_key=PORCUPINE_ACCESS_KEY,
#                 keyword_paths=[WAKE_WORD_PATH]
#             )
#             self.py_audio = pyaudio.PyAudio()
#             self.audio_stream = self.py_audio.open(
#                 rate=self.porcupine.sample_rate,
#                 channels=1,
#                 format=pyaudio.paInt16,
#                 input=True,
#                 frames_per_buffer=self.porcupine.frame_length,
#                 input_device_index=None  # Use default device
#             )
#         except Exception as e:
#             print(f"Wake detector initialization error: {e}")
#             exit(1)

#     def speak(self, text):
#         """Reliable speech using pyttsx3 (init & stop each time)"""
#         try:
#             engine = pyttsx3.init()
#             engine.setProperty('rate', 180)   # voice speed
#             engine.setProperty('volume', 1.0)  # volume
#             engine.say(text)
#             engine.runAndWait()
#             engine.stop()  # important: release engine
#         except Exception as e:
#             print(f"Speech error: {e}")

#     def listen(self):
#         """Robust listening with multiple retries"""
#         for attempt in range(3):  # Try 3 times
#             with sr.Microphone() as source:
#                 print("\n🔴 Listening...", end='', flush=True)
#                 try:
#                     self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
#                     audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
#                     print("\r🟢 Processing...", end='', flush=True)
#                     text = self.recognizer.recognize_google(audio).lower()
#                     print(f"\rYou: {text}")
#                     return text
#                 except sr.UnknownValueError:
#                     print("\r🔴 Didn't catch that", end='', flush=True)
#                     self.speak("Could you repeat that?")
#                 except Exception as e:
#                     print(f"\rError: {e}", end='', flush=True)
#         return ""  # Return empty if all attempts fail

#     def wait_for_wake_word(self):
#         """Wait for wake word with visual feedback"""
#         print("\n💤 Sleeping... Say 'Hey Pico'", end='', flush=True)
#         while True:
#             try:
#                 pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
#                 pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
#                 if self.porcupine.process(pcm) >= 0:
#                     print("\r🎤 Wake word detected!".ljust(50))
#                     self.speak("Yes? I'm listening")
#                     return True
#             except Exception as e:
#                 print(f"\rWake word error: {e}", end='', flush=True)
#                 time.sleep(0.1)

#     def generate_response(self, user_input):
#         """Generate thoughtful, contextual response"""
#         try:
#             # Add to conversation history
#             self.conversation_history.append({"role": "user", "content": user_input})
            
#             # Generate response
#             inputs = self.tokenizer([user_input], return_tensors="pt", truncation=True, max_length=128)
#             reply_ids = self.model.generate(**inputs, max_length=200)
#             response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
            
#             # Maintain conversation context
#             self.conversation_history.append({"role": "assistant", "content": response})
#             return response
            
#         except Exception as e:
#             print(f"Response error: {e}")
#             return "Let's talk about something else."

#     def conversation_loop(self):
#         """Main interactive conversation"""
#         self.speak("Hi friend! I'm Pico. What's on your mind today?")
        
#         while True:
#             user_input = self.listen()
#             if not user_input:
#                 continue
                
#             # Exit conditions
#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break
                
#             # Generate and speak response
#             response = self.generate_response(user_input)
#             print(f"Pico: {response}")
#             self.speak(response)

#     def cleanup(self):
#         """Clean up all resources"""
#         print("\n🧹 Cleaning up resources...")
#         if hasattr(self, 'audio_stream'):
#             self.audio_stream.close()
#         if hasattr(self, 'py_audio'):
#             self.py_audio.terminate()
#         if hasattr(self, 'porcupine'):
#             self.porcupine.delete()

#     def run(self):
#         """Main execution loop"""
#         try:
#             print("=== Pico AI Assistant ===")
#             print("Initializing... Please wait")
#             self.speak("Pico is ready!")
            
#             while True:
#                 if self.wait_for_wake_word():
#                     self.conversation_loop()
                    
#         except KeyboardInterrupt:
#             print("\nShutting down gracefully...")
#         except Exception as e:
#             print(f"Fatal error: {e}")
#         finally:
#             self.cleanup()

# if __name__ == "__main__":
#     assistant = PicoAssistant()
#     assistant.run()



from core.assistant import PicoAssistant
from core.AddData import  AddData
if __name__ == "__main__":
    AddData()
    assistant = PicoAssistant()
    assistant.run()



