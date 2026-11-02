
import os, time, shutil, subprocess, textwrap
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from scipy.io import wavfile

USE_MODELS = True
OUT = "exp10_outputs"; os.makedirs(OUT, exist_ok=True)
W, H, FPS, DURATION = 640, 360, 24, 20             # video: 640x360, 24 fps, 20 seconds (task: 15-30 s)
SR = 32000                                         # audio sample rate

# =====================================================================
# TASK 2 : common prompt (same theme for image and music)
# =====================================================================
IMAGE_PROMPT = ("Human genetic diversity and responsible AI in healthcare, respectful illustration of diverse human faces, "
                "abstract DNA double helix and subtle AI circuit elements, futuristic blue and purple style, calm hopeful atmosphere")
MUSIC_PROMPT = ("Gentle inspirational cinematic instrumental music about scientific discovery and human diversity, "
                "soft piano, ambient synthesizers, gradual uplifting progression, calm hopeful mood")
log = {"image_prompt": IMAGE_PROMPT, "music_prompt": MUSIC_PROMPT}

# =====================================================================
# TASK 3 : image
# =====================================================================
def procedural_image(w=1024, h=576):
    """stylised fallback: blue-purple gradient + DNA helix + diverse faces (circles with different skin tones)"""
    x = np.linspace(0, 1, w)[None, :, None]; y = np.linspace(0, 1, h)[:, None, None]
    base = np.concatenate([20 + 60 * x + 0 * y, 20 + 30 * y + 0 * x, 90 + 120 * (1 - y) + 0 * x], axis=2)   # blue -> purple
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(img, "RGBA")
    for i in range(0, w, 6):                                                    # DNA double helix
        a, b = h / 2 + 120 * np.sin(i / 55), h / 2 - 120 * np.sin(i / 55)
        d.ellipse([i - 3, a - 3, i + 3, a + 3], fill=(120, 220, 255, 220)); d.ellipse([i - 3, b - 3, i + 3, b + 3], fill=(230, 140, 255, 220))
        if i % 30 == 0: d.line([i, a, i, b], fill=(255, 255, 255, 90), width=1)
    skins = [(255, 224, 189), (234, 192, 134), (198, 134, 66), (141, 85, 36), (96, 57, 19), (241, 194, 125)]
    for k, s in enumerate(skins):                                               # row of diverse stylised faces
        cx, cy, r = 110 + k * 160, h - 120, 55
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=s + (255,), outline=(255, 255, 255, 200), width=3)
        d.ellipse([cx - 20, cy - 12, cx - 10, cy - 2], fill=(40, 30, 30, 255)); d.ellipse([cx + 10, cy - 12, cx + 20, cy - 2], fill=(40, 30, 30, 255))
        d.arc([cx - 22, cy + 5, cx + 22, cy + 30], 20, 160, fill=(80, 40, 40, 255), width=3)
    return img.filter(ImageFilter.GaussianBlur(0.8))

t0 = time.time(); image_source = "procedural fallback"
img = None
if USE_MODELS:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32)
        img = pipe(IMAGE_PROMPT, num_inference_steps=1, guidance_scale=0.0, height=512, width=512).images[0]
        image_source = "stabilityai/sd-turbo"
    except Exception as e:
        print("[image] SD-Turbo unavailable ->", type(e).__name__)
if img is None:
    img = procedural_image()
img.save(f"{OUT}/image.png")
log.update(image_model=image_source, image_size=img.size, image_time=round(time.time() - t0, 1))
print("IMAGE:", image_source, img.size, f"{log['image_time']}s")

# =====================================================================
# TASK 4 : music
# =====================================================================
def synth_music(seconds):
    """fallback: slow chord progression (Am - F - C - G): soft sine chords + a plucked arpeggio, with fade in/out"""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    chords = [[220, 261.6, 329.6], [174.6, 220, 261.6], [261.6, 329.6, 392], [196, 246.9, 293.7]]
    audio = np.zeros_like(t); seg = seconds / 8                                    # 8 segments, 4 chords repeated twice
    for i in range(8):
        notes = chords[i % 4]
        start = int(i * seg * SR); end = int((i + 1) * seg * SR)
        tt = t[start:end] - i * seg                                                  # time inside this segment
        env = np.minimum(tt / 0.8, 1) * np.exp(-0.15 * tt)                           # soft attack, slow decay
        for f in notes:                                                              # chord = 3 sine waves
            audio[start:end] += 0.18 * env * np.sin(2 * np.pi * f * tt)
        beat = (tt * 2).astype(int)                                                  # a new arpeggio note every 0.5 s
        local = tt - beat / 2                                                        # time since that note started
        freq = np.array(notes)[beat % 3] * 2                                         # one octave higher
        audio[start:end] += 0.10 * np.sin(2 * np.pi * freq * local) * np.exp(-4 * local)
    audio *= np.minimum(t / 1.5, 1) * np.minimum((seconds - t) / 2.0, 1)            # fade in / fade out
    return (audio / np.abs(audio).max() * 0.8).astype(np.float32)

t0 = time.time(); music_source = "NumPy synthesiser (fallback)"; audio = None
if USE_MODELS:
    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
        mg = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        out = mg.generate(**proc(text=[MUSIC_PROMPT], padding=True, return_tensors="pt"), do_sample=True, max_new_tokens=500)  # ~10 s
        clip = out[0, 0].detach().numpy(); sr_gen = mg.config.audio_encoder.sampling_rate
        clip = np.interp(np.linspace(0, len(clip), int(len(clip) * SR / sr_gen)), np.arange(len(clip)), clip)  # resample
        reps = int(np.ceil(DURATION * SR / len(clip)))                                                         # loop to cover the video
        audio = np.tile(clip, reps)[:DURATION * SR].astype(np.float32); music_source = "facebook/musicgen-small"
    except Exception as e:
        print("[music] MusicGen unavailable ->", type(e).__name__)
if audio is None:
    audio = synth_music(DURATION)
audio = audio * 0.7                                                                  # adjust audio level
wavfile.write(f"{OUT}/music.wav", SR, (audio * 32767).astype(np.int16))
log.update(music_model=music_source, music_seconds=len(audio) / SR, music_time=round(time.time() - t0, 1))
print("MUSIC:", music_source, f"{log['music_seconds']:.0f}s", f"{log['music_time']}s")

# =====================================================================
# TASK 5 : video  (slow zoom + pan, fades, text overlays)  -> cv2, then FFmpeg adds audio
# =====================================================================
frame_src = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
ih, iw = frame_src.shape[:2]

TEXTS = [  # (start s, end s, text, font scale)
    (1, 6, "Human Genetic Diversity and Responsible AI", 1.0),
    (7, 12, "Every individual is unique.", 1.0),
    (13, 19, "AI can support research, but clinical diagnosis requires qualified professionals.", 0.8),
]
def alpha_at(t, s, e, fade=0.8):                                  # fade in / fade out of a text block
    return float(np.clip(min(t - s, e - t) / fade, 0, 1))

def make_frame(n):
    t = n / FPS; p = n / (FPS * DURATION)                         # time and progress 0..1
    zoom = 1.0 + 0.25 * p                                         # slow zoom 1.0 -> 1.25
    cw, ch = iw / zoom, ih / zoom                                 # crop size shrinks => zoom in
    cx0 = (iw - cw) * p                                           # slow pan left -> right
    cy0 = (ih - ch) / 2
    crop = frame_src[int(cy0):int(cy0 + ch), int(cx0):int(cx0 + cw)]
    frame = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    for s, e, text, scale in TEXTS:
        a = alpha_at(t, s, e)
        if a <= 0: continue
        lines = textwrap.wrap(text, 38 if scale >= 1 else 46)
        overlay = frame.copy(); y0 = H // 2 - 20 * len(lines)
        cv2.rectangle(overlay, (0, y0 - 35), (W, y0 + 45 * len(lines)), (0, 0, 0), -1)                # dark band => readable
        frame = cv2.addWeighted(overlay, 0.45 * a, frame, 1 - 0.45 * a, 0)
        for i, line in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, 2)
            txt = frame.copy()
            cv2.putText(txt, line, ((W - tw) // 2, y0 + 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            frame = cv2.addWeighted(txt, a, frame, 1 - a, 0)
    cv2.putText(frame, "AI-generated content", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)  # disclosure
    fade = np.clip(min(t, DURATION - t) / 1.0, 0, 1)                                                  # fade from / to black
    return (frame * fade).astype(np.uint8)

silent = f"{OUT}/silent.mp4"
vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for n in range(FPS * DURATION):
    vw.write(make_frame(n))
vw.release()
cv2.imwrite(f"{OUT}/preview_frame.png", make_frame(FPS * 9))      # one frame to look at

# merge audio + video with FFmpeg
ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    try:
        import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = None
final = f"{OUT}/final_video.mp4"
if ffmpeg:
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", f"{OUT}/music.wav", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final], check=True)
    print("VIDEO saved:", final)
else:
    shutil.copy(silent, final); print("FFmpeg not found -> saved video WITHOUT audio. (pip install imageio-ffmpeg)")

# =====================================================================
# TASK 6 / 7 : record + evaluation checklist
# =====================================================================
print("\n=== RECORD ===")
for k, v in log.items(): print(f"{k}: {v}")
print(f"video: {W}x{H} @ {FPS} fps, {DURATION}s -> {final}")
print("""
=== EVALUATION CHECKLIST (answer in your own words) ===
a) Image matches prompt?            b) Music conveys the mood?          c) Audio-visual consistency?
d) Text / transitions / timing OK?  e) Artefacts (distorted faces, audio glitches)?
f) Representation respectful, no medical claims?   Disclose: content is AI-generated; check model licences.""")
