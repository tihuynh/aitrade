
# ==========================================
# 📦 Cài đặt thư viện cần thiết
# ==========================================
!pip install moviepy gtts openai python-dotenv

# ==========================================
# 📁 Upload file .env chứa API Key + nhạc nền
# ==========================================
from google.colab import files
print("⬆️ Vui lòng upload file .env và 1 file nhạc nền (mp3)")
uploaded = files.upload()

# ==========================================
# 🔐 Load OpenAI API Key từ .env
# ==========================================
import os
from dotenv import load_dotenv
load_dotenv()
import openai

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith(("sk-", "sk-proj-")):
    raise ValueError("❌ API Key không hợp lệ. Vui lòng kiểm tra lại file .env")


openai.api_key = api_key
print("✅ Đã tải OpenAI API Key thành công!")

# ==========================================
# 📁 Tạo thư mục output
# ==========================================
os.makedirs("output", exist_ok=True)

# ==========================================
# 🎨 Tạo ảnh nền bằng DALL·E
# ==========================================
import requests
from PIL import Image

def generate_image(prompt, image_path="output/image.png"):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        image_data = requests.get(image_url).content
        with open(image_path, "wb") as f:
            f.write(image_data)
        print("✅ Đã tạo ảnh AI:", image_path)
        return image_path
    except Exception as e:
        print("❌ Lỗi khi tạo ảnh:", e)
        return None

# ==========================================
# 🗣️ Tạo giọng nói bằng gTTS
# ==========================================
from gtts import gTTS

def create_tts(text, audio_path="output/voice.mp3"):
    try:
        tts = gTTS(text, lang='vi')
        tts.save(audio_path)
        print("✅ Đã tạo giọng nói:", audio_path)
        return audio_path
    except Exception as e:
        print("❌ Lỗi TTS:", e)
        return None

# ==========================================
# 🎞️ Tạo video bằng MoviePy + nhạc nền
# ==========================================
from moviepy.editor import *

def create_video(image_path, voice_path, bgm_path, script, output_path="output/final_video.mp4"):
    try:
        # Load ảnh nền
        img_clip = ImageClip(image_path).set_duration(15).resize(height=1920).set_position("center").resize(width=1080)

        # Load âm thanh
        voice_clip = AudioFileClip(voice_path)
        bgm_clip = AudioFileClip(bgm_path).volumex(0.1).set_duration(15)
        final_audio = CompositeAudioClip([bgm_clip, voice_clip])

        # Tách câu để tạo caption động
        lines = script.strip().split("\n")
        caption_clips = []
        start = 0
        duration_per_line = 15 / len(lines)

        for i, line in enumerate(lines):
            txt = TextClip(line, fontsize=70, font="DejaVu-Sans", color='white',
                           size=(1000, None), method='caption', align='South')
            txt = txt.set_start(start).set_duration(duration_per_line)
            caption_clips.append(txt)
            start += duration_per_line

        # Kết hợp các lớp
        video = CompositeVideoClip([img_clip, *caption_clips])
        video = video.set_audio(final_audio)

        # Xuất video
        video.write_videofile(output_path, fps=24, preset='ultrafast')
        print("✅ Đã xuất video:", output_path)

        # Tự động tải về máy
        from google.colab import files
        files.download(output_path)

    except Exception as e:
        print("❌ Lỗi video:", e)

# ==========================================
# ▶️ Chạy thử
# ==========================================
script = """Bạn có từng thấy mình vui... rồi buồn chỉ vì nhìn người khác?
Thì ra, mình đã để sự so sánh đánh cắp hạnh phúc."""
prompt = "a cute green sprout character sitting under a tree, reading a book, cinematic, soft lighting, Pixar style"

img_path = generate_image(prompt)
voice_path = create_tts(script)

# Tìm tên file nhạc nền vừa upload (mp3)
bgm_file = next((f for f in uploaded if f.endswith(".mp3")), None)

if img_path and voice_path and bgm_file:
    create_video(img_path, voice_path, bgm_file, script)
else:
    print("❌ Thiếu ảnh / voice / nhạc nền!")
