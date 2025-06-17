# این فایل کد اصلی برنامه است - نسخه رایگان با Hugging Face
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
# برای نوشتن متن (یک مدل قدرتمند و رایگان)
WRITER_MODEL_HF = "google/gemma-7b-it"
# برای ویرایش متن (می‌توانیم از همان مدل نویسنده استفاده کنیم)
EDITOR_MODEL_HF = "google/gemma-7b-it"

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

# <<< تغییر: این تابع جایگزین تمام مدل‌ها شده است
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
             payload = {"inputs": prompt, "options": {"wait_for_model": True}, "parameters": {"return_full_text": False, "max_new_tokens": 1024}}

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
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
        
    summary = call_huggingface_model(SUMMARIZER_MODEL_HF, search_results[:4000])

    writer_prompt = f"شما یک نویسنده متخصص علم و فناوری به زبان فارسی هستید. بر اساس این خلاصه، یک پست جذاب و خوانا برای یک کانال تلگرامی در مورد '{RESEARCH_TOPIC}' بنویس:\n\n{summary}"
    initial_post = call_huggingface_model(WRITER_MODEL_HF, writer_prompt)

    editor_prompt = f"شما یک ویراستار دقیق به زبان فارسی هستید. این متن را بازبینی و روان‌تر کن و اگر نیاز بود، اشتباهاتش را اصلاح کن. خروجی شما فقط باید متن نهایی و آماده انتشار باشد:\n\n{initial_post}"
    final_post = call_huggingface_model(EDITOR_MODEL_HF, editor_prompt)
    
    # گاهی مدل gemma پرامپت را در خروجی تکرار می‌کند، این کد آن را تمیز می‌کند
    if final_post.strip().startswith(editor_prompt.strip()):
        final_post = final_post.replace(editor_prompt, "").strip()

    final_telegram_message = f"**{RESEARCH_TOPIC}**\n\n{final_post}\n\n#هوش_مصنوعی #تکنولوژی #علم"
    await send_to_telegram(final_telegram_message)

if __name__ == "__main__":
    asyncio.run(main())
