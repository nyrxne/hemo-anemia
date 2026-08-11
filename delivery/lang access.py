import argparse
from gtts import gTTS
import os
import GoogleTranslator
TEXT = (
    "Screening result: signs consistent with possible anemia were detected. "
    "This is a research prototype, not a medical diagnosis. "
    "Please consult a doctor for a blood test to confirm."
)

LANG_MAP = {
    "hindi":   {"translate": "hi", "tts": "hi"},
    "tamil":   {"translate": "ta", "tts": "ta"},
    "bengali": {"translate": "bn", "tts": "bn"},
    "telugu":  {"translate": "te", "tts": "te"},
    "marathi": {"translate": "mr", "tts": "mr"},
}
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hindi", choices=LANG_MAP.keys())
    args = parser.parse_args()
    codes = LANG_MAP[args.language]
 
    print(f"Source (English): {TEXT}")
 
    translated = GoogleTranslator(source="en", target=codes["translate"]).translate(TEXT)
    print(f"Translated ({args.language}): {translated}")
 
    tts = gTTS(text=translated, lang=codes["tts"])
    out_path = f"hema_report_{args.language}.mp3"
    tts.save(out_path)
    print (f"Saved audio to {out_path}")
 