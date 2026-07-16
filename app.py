from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# تفعيل الـ CORS للسماح لموقعك (الواجهة) بالتحدث مع هذا السيرفر بأمان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # يسمح لأي موقع بالاتصال، سنخصصها لاحقاً
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. قراءة الكتاب المنهجي
def read_book():
    if os.path.exists("book.txt"):
        with open("book.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "ملف الكتاب غير موجود."

BOOK_CONTENT = read_book()

# بنية البيانات التي سيرسلها الطالب (السؤال والنمط)
class QuestionRequest(BaseModel):
    question: str
    mode: str

# الرابط السريع والمجاني لنموذج Qwen الذكي جداً باللغة العربية
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

@app.post("/ask")
def ask_ai(data: QuestionRequest):
    # تجهيز التوجيهات بناءً على النمط الذي يختاره الطالب
    if data.mode == "عامية":
        system_instruction = (
            f"أنت أستاذ عراقي عبقري وبسيط ومحبوب. "
            f"مهمتك هي قراءة النص المنهجي التالي:\n"
            f"\"\"\"\n{BOOK_CONTENT}\n\"\"\"\n"
            f"ثم شرح الفكرة للطالب بالهجة العراقية (العامية المحببة) وبشكل مبسط وقصير جداً "
            f"مع الحفاظ التام على الحقائق العلمية دون تحريف."
        )
    elif data.mode == "وزاري":
        system_instruction = (
            f"أنت مصحح امتحانات وزاري عراقي. "
            f"اقرأ النص المنهجي التالي:\n"
            f"\"\"\"\n{BOOK_CONTENT}\n\"\"\"\n"
            f"وقم بصياغة سؤال وزاري حقيقي للطالب بناءً عليه، وانتظر منه الإجابة لتقوم بتصحيحه لاحقاً."
        )
    else: # فصحى
        system_instruction = (
            f"أنت مساعد تعليمي ذكي لطلاب العراق. "
            f"أجب على سؤال الطالب بالاعتماد الحصري على النص التالي من الكتاب المنهجي:\n"
            f"\"\"\"\n{BOOK_CONTENT}\n\"\"\"\n"
            f"لا تخرج عن هذا النص، واشرحه بأسلوب نقاط مبسط وبفصحى مفهومة."
        )

    # دمج التوجيه مع سؤال الطالب وإرساله للنموذج عبر الـ API السريع
    payload = {
        "inputs": f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{data.question}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": 500, "temperature": 0.7}
    }
    
    try:
        response = requests.post(HF_API_URL, json=payload)
        response_json = response.json()
        
        # استخراج النص الصافي من النتيجة
        if isinstance(response_json, list) and len(response_json) > 0:
            full_text = response_json[0].get("generated_text", "")
            # تنظيف النص لعرض الجواب فقط للطالب
            answer = full_text.split("<|im_start|>assistant\n")[-1].strip()
            return {"answer": answer}
        else:
            return {"answer": "السيرفر يمر بضغط حالياً، يرجى إعادة المحاولة."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "سيرفر التعليم الذكي يعمل بنجاح وبشكل مجاني!"}
