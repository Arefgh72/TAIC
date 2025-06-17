# این فایل کد اصلی برنامه است - نسخه نهایی با مدل رایگان Mistral
import os
import requests
import asyncio
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from telegram import Bot

# موضوعی که می‌خواهید هوش مصنوعی در مورد آن تحقیق کند
RESEARCH_TOPIC = "اخبار روز ایران"

# --- مدل‌های هوش مصنوعی از HuggingFace (پلن رایگان) ---
# برای خلاصه‌سازی
SUMMARIZER_MODEL_HF = "facebook/bart-large-cnn"
# <<< اصلاح شد: مدل نویسنده و ویراستار به Mistral تغییر کرد
# برای نوشتن و ویرایش متن (یک مدل بسیار قوی و در دسترس)
GENERATIVE_MODEL_HF = "mistralai/Mistral-7B-Instruct-v0.2"


# خواندن کلیدهای API از متغیرهای محیطی گیت‌هاب
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

def research(topic: str, num_results: int = 5) -> str:
    print(f"🕵️  مامور محقق: در حال جستجو در مورد '{topic}'...")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(topic, max_results=num_results, region='wt-wt')]
            content = " ".join([res.get('body', '') for res in results])
            print("✅ تحقیق با موفقیت انجام شد.")
            return content
    except Exception as e:
        print(f"❌ خطا در حین جستجو: {e}")
        return ""

def call_huggingface_model(model_name: str, prompt: str) -> str:
    """یک مدل را از طریق Hugging Face API فراخوانی می‌کند."""
    print(f"🤗 در حال فراخوانی مدل Hugging Face: {model_name}...")
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    
    try:
        # برای مدل‌های مختلف، فرمت payload ممکن است کمی متفاوت باشد
        if "bart-large-cnn" in model_name:
             payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        else:
             # برای مدل‌های تولید متن مثل Mistral
             payload = {"inputs": prompt, "options": {"wait_for_model": True}, "parameters": {"return_full_text": False, "max_new_tokens": 1024}}

        response = requests.post(api_url, headers=headers, json=payload, timeout=120) # افزایش زمان انتظار
        response.raise_for_status() # این خط در صورت وجود خطا، برنامه را متوقف می‌کند
        data = response.json()
        
        # استخراج متن تولید شده بر اساس فرمت پاسخ مدل
        if isinstance(data, list) and data:
            if "summary_text" in data[0]:
                generated_text = data[0]['summary_text']
            elif "generated_text" in data[0]:
                generated_text = data[0]['generated_text']
            else:
                generated_text = str(data)
        else:
            generated_text = "پاسخی از مدل دریافت نشد."

        print("✅ پاسخ از Hugging Face دریافت شد.")
        return generated_text
    except Exception as e:
        print(f"❌ خطای Hugging Face: {e}")
        return f"خطا در ارتباط با مدل {model_name}"


async def send_to_telegram(message: str):
    print("📤 مامور ناشر: در حال ارسال پست به تلگرام...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        print("❌ توکن ربات یا شناسه کانال تلگرام تعریف نشده است.")
        return
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=message, parse_mode='Markdown')
        print("✅ پست با موفقیت در کانال تلگرام منتشر شد!")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")

async def main():
    search_results = research(RESEARCH_TOPIC)
    if not search_results:
        print("تحقیق ناموفق بود. برنامه متوقف شد.")
        return
        
    summary = call_huggingface_model(SUMMARIZER_MODEL_HF, search_results[:3000]) # کاهش حجم متن برای پایداری بیشتر

    writer_prompt = f"You are an expert technology and news writer in Persian. Write a clear, engaging, and accurate post for a Telegram channel about '{RESEARCH_TOPIC}' based on the following summary. Use short paragraphs and simple language. The output must be only the final Persian text of the post.\n\nSummary:\n{summary}"
    initial_post = call_huggingface_model(GENERATIVE_MODEL_HF, writer_prompt)

    editor_prompt = f"You are a strict Persian editor. Review and polish the following text. Make it more fluent and correct any grammatical or factual errors. Your output must be only the final, ready-to-publish Persian text.\n\nText to edit:\n{initial_post}"
    final_post = call_huggingface_model(GENERATIVE_MODEL_HF, editor_prompt)

    final_telegram_message = f"**{RESEARCH_TOPIC}**\n\n{final_post}\n\n#هوش_مصنوعی #خبر #ایران"
    await send_to_telegram(final_telegram_message)

if __name__ == "__main__":
    asyncio.run(main())
