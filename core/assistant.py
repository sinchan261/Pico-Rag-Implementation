# import atexit
# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine

# class PicoAssistant:
#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         atexit.register(self.cleanup)

#     def conversation_loop(self):
#         self.speech.speak("Hi friend! I'm Pico. What's on your mind today?")
#         while True:
#             user_input = self.speech.listen()
#             if not user_input:
#                 continue
#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.speech.speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break
#             response = self.conversation.generate(user_input)
#             print(f"Pico: {response}")
#             self.speech.speak(response)

#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing... Please wait")
#         self.speech.speak("Pico is ready!")
#         while True:
#             if self.wake.detect():
#                 self.speech.speak("Yes? I'm listening")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\n🧹 Cleaning up resources...")
#         self.wake.cleanup()



# import atexit
# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine
# from core.datetime_info import get_time, get_date, get_day, get_day_and_date  # new import

# class PicoAssistant:
#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         atexit.register(self.cleanup)

#     def conversation_loop(self):
#         self.speech.speak("Hi friend! I'm Pico. What's on your mind today?")
#         while True:
#             user_input = self.speech.listen()
#             if not user_input:
#                 continue

#             # ✅ If user asks for date/day
#             if "today's date" in user_input or "day" in user_input:
#                 today = get_day_and_date()
#                 self.speech.speak(f"Today is {today}")
#                 continue
#             if "time now" in user_input:
#                 today = get_time()
#                 self.speech.speak(f"Now it's {today}")
#                 continue

#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.speech.speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break

#             response = self.conversation.generate(user_input)
#             print(f"Pico: {response}")
#             self.speech.speak(response)

#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing... Please wait")
#         self.speech.speak("Pico is ready!")
#         while True:
#             if self.wake.detect():
#                 self.speech.speak("Yes? I'm listening")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\n🧹 Cleaning up resources...")
#         self.wake.cleanup()


# import atexit
# import threading
# import time
# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine
# from core.datetime_info import get_time, get_date, get_day, get_day_and_date

# class PicoAssistant:
#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         atexit.register(self.cleanup)

#     def safe_speak(self, text):
#         """Safe speak method that handles TTS errors"""
#         try:
#             self.speech.speak(text)
#         except Exception as e:
#             print(f"⚠️ TTS Error: {e}")
#             print(f"Pico (text only): {text}")  # Fallback to text output

#     def conversation_loop(self):
#         self.safe_speak("Hi friend! I'm Pico. What's on your mind today?")
#         while True:
#             user_input = self.speech.listen()
#             if not user_input:
#                 continue

#             # ✅ If user asks for date/day
#             if "today's date" in user_input or "day" in user_input:
#                 today = get_day_and_date()
#                 self.safe_speak(f"Today is {today}")
#                 continue
#             if "time now" in user_input:
#                 today = get_time()
#                 self.safe_speak(f"Now it's {today}")
#                 continue

#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.safe_speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break

#             # Start speaking immediately while generating response
#             self.safe_speak("Hmm, let me think...")
            
#             # Generate response in background thread
#             response = [None]  # Use list to store result across threads
            
#             def generate_response():
#                 response[0] = self.conversation.generate(user_input)
            
#             # Start generation in background
#             gen_thread = threading.Thread(target=generate_response, daemon=True)
#             gen_thread.start()
            
#             # Wait for generation to complete
#             gen_thread.join()
            
#             print(f"Pico: {response[0]}")
#             self.safe_speak(response[0])

#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing... Please wait")
#         self.safe_speak("Pico is ready!")
#         while True:
#             if self.wake.detect():
#                 self.safe_speak("Yes? I'm listening")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\n🧹 Cleaning up resources...")
#         self.wake.cleanup()


# import atexit
# import threading
# import time
# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine
# from core.datetime_info import get_time, get_day_and_date
# from core.image_engine import ImageEngine   # ✅ added

# class PicoAssistant:
#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         self.image_engine = ImageEngine()   # ✅ image engine
#         self.waiting_for_image_prompt = False
#         self.waiting_for_image_size = False
#         self.temp_image_prompt = None
#         atexit.register(self.cleanup)

#     def safe_speak(self, text):
#         """Safe speak method that handles TTS errors"""
#         try:
#             self.speech.speak(text)
#         except Exception as e:
#             print(f"⚠️ TTS Error: {e}")
#             print(f"Pico (text only): {text}")  # fallback

#     def handle_command(self, user_input: str) -> str:
#         """Handle commands including image generation"""
#         command = user_input.lower()

#         # 🎨 --- IMAGE SECTION ---
#         if "generate image" in command:
#             self.waiting_for_image_prompt = True
#             return "I can generate an image for you! Please tell me what image you want."

#         if self.waiting_for_image_prompt and not self.waiting_for_image_size:
#             self.temp_image_prompt = command
#             self.waiting_for_image_prompt = False
#             self.waiting_for_image_size = True
#             return "Great! What size do you want for your image? (e.g., 256x256, 512x512, 1024x1024)"

#         if self.waiting_for_image_size:
#             size = command.strip().lower()
#             self.waiting_for_image_size = False
#             prompt = self.temp_image_prompt
#             file_path = self.image_engine.generate_image(prompt, size)
#             return f"Your image has been generated and saved here: {file_path}"

#         # ✅ Date / Time
#         if "today's date" in command or "day" in command:
#             return f"Today is {get_day_and_date()}"

#         if "time now" in command:
#             return f"Now it's {get_time()}"

#         if "hello" in command:
#             return "Hi! I’m Pico. How can I help you today?"

#         return None  # let conversation engine handle it

#     def conversation_loop(self):
#         self.safe_speak("Hi friend! I'm Pico. What's on your mind today?")
#         while True:
#             user_input = self.speech.listen()
#             if not user_input:
#                 continue

#             # check image + commands
#             response = self.handle_command(user_input)
#             if response:
#                 print(f"Pico: {response}")
#                 self.safe_speak(response)
#                 continue

#             # exit
#             if any(word in user_input for word in ["bye", "goodbye", "exit", "stop"]):
#                 self.safe_speak("It was great chatting! Just say 'Hey Pico' when you want to talk again.")
#                 break

#             # fallback to conversation engine
#             self.safe_speak("Hmm, let me think...")
#             response = [None]

#             def generate_response():
#                 response[0] = self.conversation.generate(user_input)

#             gen_thread = threading.Thread(target=generate_response, daemon=True)
#             gen_thread.start()
#             gen_thread.join()

#             print(f"Pico: {response[0]}")
#             self.safe_speak(response[0])

#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing... Please wait")
#         self.safe_speak("Pico is ready!")
#         while True:
#             if self.wake.detect():
#                 self.safe_speak("Yes? I'm listening")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\n🧹 Cleaning up resources...")    new unlock
#         self.wake.cleanup()



# # core/assistant.py
# import atexit
# import threading
# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine
# from core.datetime_info import get_time, get_day_and_date
# from core.image_engine import ImageEngine
# from utils.task_bus import TaskBus


# class PicoAssistant:
#     """
#     Multitasking Pico:
#     - Conversation runs in foreground (LLM).
#     - Commands dispatch background tasks via TaskBus.
#     - Results are announced only when Pico finishes speaking.
#     """

#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         self.image_engine = ImageEngine()
#         self.bus = TaskBus()

#         # track active tasks for possible cancellation
#         self.active_tasks = {}

#         atexit.register(self.cleanup)

#     # ---------- helpers ----------
#     def safe_speak(self, text: str):
#         """Speak safely, fallback to print if TTS fails"""
#         try:
#             self.speech.speak(text)
#         except Exception as e:
#             print(f"⚠️ TTS Error: {e}")
#             print(f"Pico (text only): {text}")

#     def _deliver_background_results_if_free(self):
#         """Announce exactly ONE result when Pico is not speaking."""
#         if self.speech.is_speaking:
#             return

#         result = self.bus.get_result()
#         if result:
#             # craft spoken message
#             if result.kind == "image_done" and result.payload:
#                 msg = f"✅ Your image is ready. Saved at: {result.payload}"
#             else:
#                 msg = result.message

#             print(f"Pico (bg): {msg}")
#             self.safe_speak(msg)

#     # ---------- intent + dispatch ----------
#     def _is_cancel(self, text: str) -> bool:
#         t = text.lower()
#         cancel_tokens = [
#             "cancel", "stop", "no need", "don't want",
#             "dont want", "never mind", "nevermind",
#             "leave it", "forget it"
#         ]
#         return any(tok in t for tok in cancel_tokens)

#     def _maybe_dispatch_command(self, user_input: str) -> bool:
#         """
#         Detect command(s) and dispatch them to background if found.
#         Returns True if a command was dispatched.
#         """
#         text = user_input.lower()

#         # cancel any running image task
#         if self._is_cancel(text) and "image" in text:
#             cancelled = 0
#             for tid, kind in list(self.active_tasks.items()):
#                 if kind == "image":
#                     if self.bus.cancel(tid):
#                         cancelled += 1
#                     self.active_tasks.pop(tid, None)

#             if cancelled:
#                 self.safe_speak("❎ Okay, I’ve cancelled your image request.")
#             else:
#                 self.safe_speak("There wasn’t an image task to cancel.")
#             return True

#         # image generation
#         if "generate image" in text or "create image" in text or "draw" in text:
#             prompt = user_input
#             size = "512x512"  # default size for now

#             self.safe_speak("🎨 Got it! I’m generating your image in the background while we chat.")

#             # dispatch background job
#             task_id = self.bus.submit(
#                 self._bg_generate_image,
#                 prompt, size,
#                 kind="image_done",
#                 done_msg="✅ Your image is ready."
#             )
#             self.active_tasks[task_id] = "image"
#             return True

#         # date / time
#         if "today's date" in text or "day" in text:
#             self.safe_speak(f"Today is {get_day_and_date()}")
#             return True

#         if "time now" in text or "what time" in text:
#             self.safe_speak(f"Now it's {get_time()}")
#             return True

#         if "hello" in text:
#             self.safe_speak("Hi! I’m Pico. How can I help you today?")
#             return True

#         return False

#     # ---------- background task wrappers ----------
#     def _bg_generate_image(self, prompt: str, size: str):
#         """Run image generation in the background."""
#         return self.image_engine.generate_image(prompt, size)

#     # ---------- main conversation loop ----------
#     def conversation_loop(self):
#         self.safe_speak("Hi! I’m Pico. What’s on your mind?")
#         while True:
#             # 1) deliver any finished background result if free
#             self._deliver_background_results_if_free()

#             # 2) listen for user input
#             user_input = self.speech.listen()
#             if user_input is None:
#                 continue

#             # exit
#             if any(w in user_input.lower() for w in ["bye", "goodbye", "exit", "stop talking"]):
#                 self.safe_speak("It was great chatting! Say ‘Hey Pico’ when you want me again.")
#                 break

#             # 3) detect and dispatch commands
#             dispatched = self._maybe_dispatch_command(user_input)

#             # 4) fallback: conversational reply
#             if not dispatched:
#                 self.safe_speak("🤔 Let me think...")
#                 response = [None]

#                 def generate_response():
#                     try:
#                         response[0] = self.conversation.generate(user_input)
#                     except Exception as e:
#                         response[0] = f"⚠️ Sorry, I had trouble thinking: {e}"

#                 t = threading.Thread(target=generate_response, daemon=True)
#                 t.start()
#                 t.join(timeout=10)

#                 final = response[0] or "⚠️ I couldn’t generate a proper reply this time."
#                 print(f"Pico: {final}")
#                 self.safe_speak(final)

#             # 5) after speaking, see if any background results are ready
#             self._deliver_background_results_if_free()

#     # ---------- run / cleanup ----------
#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing... Please wait")
#         self.safe_speak("Pico is ready! Say ‘Hey Pico’ to wake me up.")
#         while True:
#             if self.wake.detect():
#                 self.safe_speak("Yes? I’m listening.")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\n🧹 Cleaning up resources...")
#         try:
#             self.wake.cleanup()
#         except Exception as e:
#             print(f"⚠️ Cleanup error: {e}")   new unlock



# # core/assistant.py
# import atexit
# import threading
# import time
# import os
# import platform
# import subprocess

# from core.speech import SpeechEngine
# from core.wake_word import WakeWordDetector
# from core.conversation import ConversationEngine
# from core.datetime_info import get_time, get_day_and_date
# from core.image_engine import ImageEngine
# from utils.task_bus import TaskBus
# from core.workers import BackgroundWorkers
# from core.dispatcher import CommandDispatcher


# class PicoAssistant:
#     """
#     Foreground: LLM conversation (responsive).
#     Background: Heavy tasks via TaskBus (image generation, etc.).
#     Results are announced only when Pico finishes speaking.
#     """

#     def __init__(self):
#         self.speech = SpeechEngine()
#         self.wake = WakeWordDetector()
#         self.conversation = ConversationEngine()
#         self.image_engine = ImageEngine()
#         self.bus = TaskBus()
#         self.active_tasks = {}  # task_id -> kind (for example, "image")
#         self.last_image_path = None  # store last generated image for "show me"

#         # background workers + dispatcher
#         self.workers = BackgroundWorkers(self.image_engine)
#         self.command_dispatcher = CommandDispatcher(self.workers, self.bus)

#         atexit.register(self.cleanup)

#     # ------------- helpers -------------
#     def safe_speak(self, text: str):
#         try:
#             self.speech.speak(text)
#         except Exception as e:
#             print(f"TTS Error: {e}")
#             print(f"Pico (text only): {text}")

#     def _show_image(self, path: str):
#         """Open an image file with the default OS viewer."""
#         try:
#             if platform.system() == "Windows":
#                 os.startfile(path)
#             elif platform.system() == "Darwin":  # macOS
#                 subprocess.run(["open", path])
#             else:  # Linux
#                 subprocess.run(["xdg-open", path])
#         except Exception as e:
#             print(f"Failed to open image {path}: {e}")

#     def _deliver_background_results_if_free(self):
#         """
#         Announce exactly one background result if Pico is not speaking.
#         """
#         if getattr(self.speech, "is_speaking", False):
#             return

#         result = self.bus.get_result()
#         if not result:
#             return

#         msg = result.message
#         if result.kind == "image_done":
#             if isinstance(result.payload, str):
#                 if "failed" in result.payload.lower():
#                     msg = result.payload
#                 else:
#                     msg = f"Your image is ready. Saved at: {result.payload}"
#                     self.last_image_path = result.payload  # store for "show me"
#             elif result.payload:
#                 msg = "Your image is ready."

#         print(f"Pico (background): {msg}")
#         self.safe_speak(msg)

#     # ------------- intent + dispatch -------------
#     def _is_cancel(self, text: str) -> bool:
#         t = text.lower()
#         tokens = [
#             "cancel", "stop", "no need", "never mind", "nevermind",
#             "leave it", "forget it", "don't want", "dont want"
#         ]
#         return any(tok in t for tok in tokens)

#     def _maybe_dispatch_command(self, user_input: str) -> bool:
#         """
#         Detect commands and dispatch background jobs.
#         Returns True if a command was dispatched.
#         """
#         text = user_input.lower()

#         # cancel image tasks
#         if self._is_cancel(text) and "image" in text:
#             cancelled = 0
#             for tid, kind in list(self.active_tasks.items()):
#                 if kind == "image":
#                     if self.bus.cancel(tid):
#                         cancelled += 1
#                     self.active_tasks.pop(tid, None)
#             if cancelled:
#                 self.safe_speak("Okay, I have cancelled your image request.")
#             else:
#                 self.safe_speak("There was no image task to cancel.")
#             return True

#         # show image
#         if "show me the image" in text or "display image" in text:
#             if self.last_image_path and os.path.exists(self.last_image_path):
#                 self.safe_speak("Here is your last generated image.")
#                 threading.Thread(
#                     target=lambda: self._show_image(self.last_image_path),
#                     daemon=True
#                 ).start()
#             else:
#                 self.safe_speak("Sorry, I don't have an image ready to show yet.")
#             return True

#         # handle image generation
#         if "generate image" in text or "create image" in text or "draw" in text:
#             self.safe_speak("Sure! Please describe the image you want me to create.")

#             # Listen for up to 10 seconds
#             description = None
#             start_time = time.time()
#             while time.time() - start_time < 10:
#                 description = self.speech.listen()
#                 if description:
#                     break

#             if not description:
#                 self.safe_speak("I didn't catch that. Please try again later.")
#                 return True

#             # Dispatch background image generation
#             task_id = self.command_dispatcher.handle(description)
#             if task_id:
#                 self.active_tasks[task_id] = "image"

#             self.safe_speak(
#                 "Got it! I'm generating your image in the background while we continue talking."
#             )
#             return True

#         # quick utilities
#         if "today's date" in text or "day" in text:
#             self.safe_speak(f"Today is {get_day_and_date()}")
#             return True

#         if "time now" in text or "what time" in text:
#             self.safe_speak(f"Now it is {get_time()}")
#             return True

#         if "hello" in text or "hi " in text or text.strip() == "hi":
#             self.safe_speak("Hi. I am Pico. How can I help you today?")
#             return True

#         return False

#     # ------------- main conversation loop -------------
#     def conversation_loop(self):
#         self.safe_speak("Hi. I am Pico. What is on your mind?")
#         while True:
#             # 1) deliver one background result if free
#             self._deliver_background_results_if_free()

#             # 2) listen
#             user_input = self.speech.listen()
#             if not user_input:
#                 continue

#             # exit
#             if any(w in user_input.lower() for w in ["bye", "goodbye", "exit", "stop talking"]):
#                 self.safe_speak("It was nice talking with you. Say Hey Pico when you want me again.")
#                 break

#             # 3) detect and dispatch background commands
#             dispatched = self._maybe_dispatch_command(user_input)

#             # 4) foreground conversation only if no command was dispatched
#             if not dispatched:
#                 self.safe_speak("Let me think.")
#                 response = [None]

#                 def think():
#                     try:
#                         response[0] = self.conversation.generate(user_input)
#                     except Exception as e:
#                         response[0] = f"Sorry, I had trouble generating a reply: {e}"

#                 t = threading.Thread(target=think, daemon=True)
#                 t.start()
#                 t.join(timeout=30)

#                 final = response[0] or "I could not generate a proper reply this time"
#                 print(f"Pico: {final}")
#                 self.safe_speak(final)

#             # 5) after speaking, deliver one background result if available
#             self._deliver_background_results_if_free()

#     # ------------- run / cleanup -------------
#     def run(self):
#         print("=== Pico AI Assistant ===")
#         print("Initializing. Please wait.")
#         self.safe_speak("Pico is ready. Say Hey Pico to wake me up.")
#         while True:
#             if self.wake.detect():
#                 self.safe_speak("Yes. I am listening.")
#                 self.conversation_loop()

#     def cleanup(self):
#         print("\nCleaning up resources.")
#         try:
#             self.wake.cleanup()
#         except Exception as e:
#             print(f"Cleanup error: {e}")



# core/assistant.py
import atexit
import threading
import time
import os
import platform
import subprocess

from core.speech import SpeechEngine
from core.wake_word import WakeWordDetector
from core.conversation import ConversationEngine
from core.datetime_info import get_time, get_day_and_date
from core.image_engine import ImageEngine
from utils.task_bus import TaskBus
from core.workers import BackgroundWorkers
from core.dispatcher import CommandDispatcher


class PicoAssistant:
    """
    Foreground: LLM conversation (responsive).
    Background: Heavy tasks via TaskBus (image generation, etc.).
    Results are announced only when Pico finishes speaking.
    """

    def __init__(self):
        self.speech = SpeechEngine()
        self.wake = WakeWordDetector()
        self.conversation = ConversationEngine()
        self.image_engine = ImageEngine()
        self.bus = TaskBus()
        self.active_tasks = {}  # task_id -> kind (for example, "image")
        self.last_image_path = None  # store last generated image for "show me"

        # background workers + dispatcher
        self.workers = BackgroundWorkers(self.image_engine)
        self.command_dispatcher = CommandDispatcher(self.workers, self.bus)

        atexit.register(self.cleanup)

    # ------------- helpers -------------
    def safe_speak(self, text: str):
        try:
            self.speech.speak(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            print(f"Pico (text only): {text}")

    def _show_image(self, path: str):
        """Open an image file with the default OS viewer."""
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", path])
            else:  # Linux
                subprocess.run(["xdg-open", path])
        except Exception as e:
            print(f"Failed to open image {path}: {e}")

    def _handle_image_reaction(self):
        """Listen for a short reaction after showing an image."""
        self.safe_speak("What do you think about this image?")
        reaction = None
        start = time.time()
        while time.time() - start < 8:  # give up to 8s for a reaction
            reaction = self.speech.listen()
            if reaction:
                break

        if not reaction:
            return  # no reaction, skip

        r = reaction.lower()
        if any(word in r for word in ["nice", "good", "amazing", "wow", "love", "beautiful"]):
            self.safe_speak("I'm glad you liked the image! 🌟 Would you like me to create another one?")
        elif any(word in r for word in ["bad", "ugly", "not good", "poor", "terrible"]):
            self.safe_speak("Oh, sorry about that 😔. Should I try generating a new version for you?")
        else:
            # fallback → treat as normal conversation
            self.safe_speak("Thanks for your feedback. Let's continue.")
            print(f"Pico (reaction fallback): {reaction}")

    def _deliver_background_results_if_free(self):
        """
        Announce exactly one background result if Pico is not speaking.
        """
        if getattr(self.speech, "is_speaking", False):
            return

        result = self.bus.get_result()
        if not result:
            return

        msg = result.message
        if result.kind == "image_done":
            if isinstance(result.payload, str):
                if "failed" in result.payload.lower():
                    msg = result.payload
                else:
                    msg = f"Your image is ready. Saved at: {result.payload}"
                    self.last_image_path = result.payload  # store for "show me"
            elif result.payload:
                msg = "Your image is ready."

        print(f"Pico (background): {msg}")
        self.safe_speak(msg)

    # ------------- intent + dispatch -------------
    def _is_cancel(self, text: str) -> bool:
        t = text.lower()
        tokens = [
            "cancel", "stop", "no need", "never mind", "nevermind",
            "leave it", "forget it", "don't want", "dont want"
        ]
        return any(tok in t for tok in tokens)

    def _maybe_dispatch_command(self, user_input: str) -> bool:
        """
        Detect commands and dispatch background jobs.
        Returns True if a command was dispatched.
        """
        text = user_input.lower()

        # cancel image tasks
        if self._is_cancel(text) and "image" in text:
            cancelled = 0
            for tid, kind in list(self.active_tasks.items()):
                if kind == "image":
                    if self.bus.cancel(tid):
                        cancelled += 1
                    self.active_tasks.pop(tid, None)
            if cancelled:
                self.safe_speak("Okay, I have cancelled your image request.")
            else:
                self.safe_speak("There was no image task to cancel.")
            return True

        # show image
        if "show me the image" in text or "display image" in text:
            if self.last_image_path and os.path.exists(self.last_image_path):
                self.safe_speak("Here is your last generated image.")
                threading.Thread(
                    target=lambda: self._show_image(self.last_image_path),
                    daemon=True
                ).start()
                # After showing, handle reaction
                self._handle_image_reaction()
            else:
                self.safe_speak("Sorry, I don't have an image ready to show yet.")
            return True

        # handle image generation
        IMAGE_KEYWORDS = [
              "generate image",
              "create image",
              "create a image",
               "draw",
             "create portrait",
            "make a sketch",
             "paint",
            "artwork",
                "sketch",
                "illustration",
]
        if any(keyword in text for keyword in IMAGE_KEYWORDS):
            self.safe_speak("Sure! Please describe the image you want me to create.")

            # Listen for up to 10 seconds
            description = None
            start_time = time.time()
            while time.time() - start_time < 10:
                description = self.speech.listen()
                if description:
                    break

            if not description:
                self.safe_speak("I didn't catch that. Please try again later.")
                return True

            # Dispatch background image generation
            task_id = self.command_dispatcher.handle(description)
            if task_id:
                self.active_tasks[task_id] = "image"

            self.safe_speak(
                "Got it! I'm generating your image in the background while we continue talking."
            )
            return True

        # quick utilities
        if "today's date" in text or "day" in text:
            self.safe_speak(f"Today is {get_day_and_date()}")
            return True

        if "time now" in text or "what time" in text:
            self.safe_speak(f"Now it is {get_time()}")
            return True

        if "hello" in text or "hi " in text or text.strip() == "hi":
            self.safe_speak("Hi. I am Pico. How can I help you today?")
            return True

        return False

    # ------------- main conversation loop -------------
    def conversation_loop(self):
        self.safe_speak("Hi. I am Pico. What is on your mind?")
        while True:
            # 1) deliver one background result if free
            self._deliver_background_results_if_free()

            # 2) listen
            user_input = self.speech.listen()
            if not user_input:
                continue

            # exit
            if any(w in user_input.lower() for w in ["stop","wait","bye", "goodbye", "exit", "stop talking"]):
                self.safe_speak("It was nice talking with you. Say Hey Pico when you want me again.")
                break

            # 3) detect and dispatch background commands
            dispatched = self._maybe_dispatch_command(user_input)

            # 4) foreground conversation only if no command was dispatched
            if not dispatched:
                self.safe_speak("Let me think.")
                response = [None]

                def think():
                    try:
                        response[0] = self.conversation.generate(user_input)
                    except Exception as e:
                        response[0] = f"Sorry, I had trouble generating a reply: {e}"

                t = threading.Thread(target=think, daemon=True)
                t.start()
                t.join(timeout=30)

                final = response[0] or "I could not generate a proper reply this time"
                print(f"Pico: {final}")
                self.safe_speak(final)

            # 5) after speaking, deliver one background result if available
            self._deliver_background_results_if_free()

    # ------------- run / cleanup -------------
    def run(self):
        print("=== Pico AI Assistant ===")
        print("Initializing. Please wait.")
        self.safe_speak("Pico is ready. Say Hey Pico to wake me up.")
        while True:
            if self.wake.detect():
                self.safe_speak("Yes. I am listening.")
                self.conversation_loop()

    def cleanup(self):
        print("\nCleaning up resources.")
        try:
            self.wake.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
