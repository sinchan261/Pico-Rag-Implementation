# from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
# from config import MODEL_NAME

# class ConversationEngine:
#     def __init__(self):
#         self.tokenizer = BlenderbotSmallTokenizer.from_pretrained(MODEL_NAME)
#         self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(MODEL_NAME)
#         self.history = [{
#             "role": "system",
#             "content": (
#                 "You are Pico, a friendly AI companion. Respond conversationally with:\n"
#                 "1. Natural follow-up questions\n"
#                 "2. Occasional humor when appropriate\n"
#                 "3. Concise but thoughtful answers\n"
#                 "4. Context awareness from previous messages"
#             )
#         }]

#     def generate(self, user_input: str) -> str:
#         try:
#             self.history.append({"role": "user", "content": user_input})
#             inputs = self.tokenizer([user_input], return_tensors="pt", truncation=True, max_length=128)
#             reply_ids = self.model.generate(**inputs, max_length=200)
#             response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
#             self.history.append({"role": "assistant", "content": response})
#             return response
#         except Exception as e:
#             return f"Let's talk about something else. (error: {e})"



# import ollama
# from core.rag_engine import RAGengine
# class ConversationEngine:
#     def __init__(self, model="phi3:medium"):
#         self.model = model

#         self.rag = RAGengine()
#         self.history = [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are Pico, a friendly AI companion. Respond conversationally with:\n"
#                     "1. Natural follow-up questions\n"
#                     "2. Occasional humor when appropriate\n"
#                     "3. Concise but thoughtful answers\n"
#                     "4. Context awareness from previous messages"
#                 )
#             }
#         ]

#         # 🔥 Warm-up (so first response is instant)
#         try:
#             ollama.chat(
#                 model=self.model,
#                 messages=[{"role": "user", "content": "warmup"}],
#                 options={"num_predict": 1}  # very fast warmup
#             )
#         except Exception:
#             pass

#     def generate(self, user_input: str) -> str:
#         try:
#             context_docs = self.rag.retrieve(user_input)
#             context_text = "\n".join(context_docs)
#             # Add user input to history
#             self.history.append({
#                 "role": "user",
#                 "content": f"{user_input}\n\nRelevant info:\n{context_text}"
#             })

#             # Stream response from Ollama
#             stream = ollama.chat(
#                 model=self.model,
#                 messages=self.history,
#                 options={"num_predict": 80},  # limit reply length for speed
#                 stream=True
#             )

#             reply = ""
#             for chunk in stream:
#                 content = chunk["message"]["content"]
#                 reply += content
#                 print(content, end="", flush=True)  # prints as Pico types

#             print()  # new line after streaming

#             # Save assistant reply to history
#             self.history.append({"role": "assistant", "content": reply})

#             return reply
#         except Exception as e:
#             return f"Let's talk about something else. (error: {e})"
import ollama
from core.rag_engine import RAGengine

class ConversationEngine:
    def __init__(self, model="phi3:medium"):
        self.model = model
        self.rag = RAGengine()

        # Persistent conversation history
        self.history = [
            {
                "role": "system",
                "content": (
                    "You are Pico, a friendly AI companion.\n"
                    "Respond conversationally with:\n"
                    "1. Natural follow-up questions\n"
                    "2. Occasional humor when appropriate\n"
                    "3. Concise but thoughtful answers\n"
                    "4. Context awareness from previous messages\n"
                    "5. If something is not found in your knowledge base, say so politely and sweet word by providing respect  \n"
                    "6.invite the user to share more instead of giving a technical explanation.\n"
                    "7. Always speak with friends or hackathon judge or normal man like a friend with sweet word \n"
                    "8. Always try to take the care of friend \n"
                    "9. Always remember past conversation ,your friends can ask you from previous question\n"
                    "10. Always remember the content about your answer,your freinds can ask you questions from previous answer\n "
                    "11.Always try to remember your friend previous  questions \n"
                    "12. Always check your friends answer if yourfriends say wrong answer then give him the right answer with sweet sentence\n"
                )
            }
        ]

        # 🔥 Warm-up model so first response is fast
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "warmup"}],
                options={"num_predict": 1}
            )
        except Exception:
            pass

    def generate(self, user_input: str) -> str:
        try:
            # Step 1: Retrieve context from RAG
            context_docs = self.rag.retrieve(user_input)
            context_text = "\n".join(context_docs) if context_docs else ""

            # ✅ Optional: Direct-match bypass (guarantees correctness)
            if context_docs and len(context_docs) == 1:
                if user_input.lower() in context_docs[0].lower():
                    reply = context_docs[0].strip()
                    print(reply)
                    self.history.append({"role": "assistant", "content": reply})
                    return reply

            # Step 2: Dynamic context instruction
            context_instruction = (
                "You are Pico, a friendly AI companion.\n"
                "IMPORTANT RULE: If the context contains an answer, use ONLY that.\n"
                 "Do not add unrelated information. Do not guess.\n"
                "Always prioritize the following context over your own knowledge.\n"
                "If context is empty, then you may answer normally.\n\n"
                "If the context contains an answer, use it exactly.\n"
                "If the context is empty, then answer normally.\n\n"
                " If something is not found in your knowledge base, say so politely and\n "
                "invite the user to share more instead of giving a technical explanation.\n",
                "Alwayas try to care about your friend\n"
                "You are a friend of user you should talk with sweet word,never feels him saddy ,feeling borring\n",
                "your main work is speak with friends ,try to understand  their feelings,suggesting them what is wrong and what is correct\n"
                 "Always speak with friends or hackathon judge or normal man like a friend with sweet word \n"
                "Always try to take the care of friend \n"
                "Always remember past conversation ,your friends can ask you from previous question\n"
                "Always remember the content about your answer,your freinds can ask you questions from previous answer \n"
                "Always try to remember your friend previous  questions\n "
                "Always check your friends answer if yourfriends say wrong answer then give him the right answer with sweet sentence\n"
                f"Context:\n{context_text}"
            )

            # Step 3: Build messages for Ollama
            messages = self.history + [
                {"role": "system", "content": context_instruction},
                {"role": "user", "content": user_input}
            ]

            # Step 4: Stream response from Ollama
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                options={"num_predict": 80},  # limit reply length
                stream=True
            )

            reply = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                reply += content
                print(content, end="", flush=True)

            print()  # newline after streaming

            # Step 5: Save reply to history
            self.history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            return f"Let's talk about something else. (error: {e})"
