# این فایل کد اصلی برنامه است
import os
import replicate
import requests
import asyncio
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from telegram import Bot

# موضوعی که می‌خواهید هوش مصنوعی در مورد آن تحقیق کند
# برای تغییر موضوع، فقط متن داخل " " را در خط بعدی عوض کنید
RESEARCH_TOPIC = "اخبار روز و مهم ایران"

# مدل‌های هوش مصنوعی
# <<< اصلاح شد: نام مدل دقیق‌تر شد
WRITER_MODEL_REPLICATE = "meta/meta-llama-3-70b-instruct"
# <<< اصلاح شد: به یک مدل قوی و قابل اعتماد تغییر کرد
EDITOR_MODEL_REPLICATE = "mistralai/mixtral-8x7b-instruct-v0.1"
SUMMARIZER_MODEL_HF = "facebook/bart-large-cnn"

# خواندن کلیدهای API از متغیرهای محیطی که در گیت‌هاب تنظیم می‌کنیم
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# تنظیم کلید Replicate
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

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

def call_replicate_model(model_name: str, system_prompt: str, user_prompt: str) -> str:
    print(f"🧠 در حال فراخوانی مدل Replicate: {model_name}...")
    try:
        output = replicate.run(
            model_name,
            input={
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": 2048
            }
        )
        result = "".join(output)
        print("✅ پاسخ از Replicate دریافت شد.")
        return result
    except Exception as e:
        print(f"❌ خطای Replicate: {e}")
        return f"خطا در ارتباط با مدل {model_name}"

def call_huggingface_summarizer(text_to_summarize: str, model_name: str) -> str:
    print(f"🤗 در حال فراخوانی مدل Hugging Face: {model_name}...")
    try:
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        
        response = requests.post(api_url, headers=headers, json={"inputs": text_to_summarize, "options": {"wait_for_model": True}})
        response.raise_for_status()
        
        summary = response.json()[0]['summary_text']
        print("✅ خلاصه از Hugging Face دریافت شد.")
        return summary
    except Exception as e:
        print(f"❌ خطای Hugging Face: {e}")
        return "خطا در خلاصه‌سازی متن."

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
        
    summary = call_huggingface_summarizer(search_results[:4000], SUMMARIZER_MODEL_HF)

    writer_system_prompt = "شما یک نویسنده متخصص علم و فناوری به زبان فارسی هستید. وظیفه شما این است که بر اساس اطلاعات داده شده، یک پست جذاب، دقیق و خوانا برای یک کانال تلگرامی بنویسید. از پاراگراف‌های کوتاه و زبان ساده استفاده کنید."
    writer_user_prompt = f"بر اساس این خلاصه، یک پست کامل در مورد '{RESEARCH_TOPIC}' بنویس: \n\n{summary}"
    initial_post = call_replicate_model(WRITER_MODEL_REPLICATE, writer_system_prompt, writer_user_prompt)

    editor_system_prompt = "شما یک ویراستار دقیق و سخت‌گیر به زبان فارسی هستید. متنی که به شما داده می‌شود را بازبینی کنید. اشتباهات گرامری و علمی را اصلاح کنید، جمله‌بندی را روان‌تر کنید و در صورت نیاز، عنوان جذاب‌تری برای آن پیشنهاد دهید. خروجی شما فقط باید متن نهایی و آماده انتشار باشد."
    editor_user_prompt = f"این متن را ویرایش و نهایی کن: \n\n{initial_post}"
    final_post = call_replicate_model(EDITOR_MODEL_REPLICATE, editor_system_prompt, editor_user_prompt)

    final_telegram_message = f"**{RESEARCH_TOPIC}**\n\n{final_post}\n\n#هوش_مصنوعی #تکنولوژی #علم"
    await send_to_telegram(final_telegram_message)

if __name__ == "__main__":
    asyncio.run(main())
