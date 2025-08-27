from deep_translator import GoogleTranslator

def translate(text: str, src: str = "auto", dest: str = "en") -> str:
    """
    Translate text from source language to destination language.
    src='auto' detects the source automatically.
    """
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception as e:
        print("Translation error:", e)
        return text
