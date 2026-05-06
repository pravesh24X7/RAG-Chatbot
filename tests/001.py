# =========================================================
# production_video_generator.py
# =========================================================
#
# Production-grade Educational Video Generator
#
# PIPELINE:
#
# RAG Answer
#    ↓
# LLM Storyboard Generation
#    ↓
# Structured Visual Planning
#    ↓
# Dynamic Slide Rendering
#    ↓
# Offline TTS Narration
#    ↓
# Subtitle Generation
#    ↓
# Final MP4 Rendering
#
# =========================================================
# INSTALL
# =========================================================
#
# pip install:
#
# pillow
# moviepy==1.0.3
# pyttsx3
# langchain
# langchain-groq
#
# =========================================================

import os
import re
import json
import time
import textwrap

from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont
)

from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips
)

from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = Path("output")
SLIDES_DIR = OUTPUT_DIR / "slides"
AUDIO_DIR = OUTPUT_DIR / "audio"

OUTPUT_DIR.mkdir(exist_ok=True)
SLIDES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

VIDEO_OUTPUT = OUTPUT_DIR / "final_explainer.mp4"

WIDTH = 1280
HEIGHT = 720

FONT_PATH = "arial.ttf"

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


# =========================================================
# FONT HELPERS
# =========================================================

def get_font(size=40):

    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()


# =========================================================
# JSON CLEANER
# =========================================================

def clean_json_response(text: str):

    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()


# =========================================================
# STORYBOARD VALIDATION
# =========================================================

def validate_scene(scene):

    required = [
        "title",
        "narration",
        "diagram_type",
        "subtitle"
    ]

    for field in required:
        if field not in scene:
            return False

    return True


# =========================================================
# FALLBACK STORYBOARD
# =========================================================

def fallback_storyboard(final_answer: str):

    paragraphs = [
        p.strip()
        for p in final_answer.split(".")
        if len(p.strip()) > 20
    ]

    scenes = []

    for idx, para in enumerate(paragraphs[:6]):

        scenes.append({
            "title": f"Concept {idx + 1}",
            "narration": para,
            "diagram_type": "text_only",
            "subtitle": para[:60]
        })

    return scenes


# =========================================================
# STORYBOARD GENERATOR
# =========================================================

def generate_storyboard(final_answer: str):

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.3
    )

    system_prompt = """
You are an expert educational storyboard creator.

Your job:
Convert educational content into a cinematic explainer video storyboard.

RULES:
- Return ONLY valid JSON
- No markdown
- No code fences
- No explanation
- Minimum 4 scenes
- Each scene explains ONE concept
- Prefer visual explanations
- Equations deserve separate scenes
"""

    user_prompt = f"""
Convert the educational content below into a storyboard.

AVAILABLE DIAGRAM TYPES:
[
"equation",
"process_flow",
"comparison",
"cycle",
"timeline",
"text_only",
"network",
"hierarchy",
"wave",
"atom",
"energy_levels",
"probability_cloud"
]

SCHEMA:

{{
"title": "string",
"narration": "2-4 sentence conversational explanation",
"diagram_type": "diagram type",
"subtitle": "3-6 words",

"equation_text": "optional",

"items": [],

"labels": [],

"left_label": "",
"right_label": "",

"left_items": [],
"right_items": []
}}

CONTENT:
{final_answer}
"""

    try:

        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        raw = response.content

        cleaned = clean_json_response(raw)

        storyboard = json.loads(cleaned)

        valid_scenes = []

        for scene in storyboard:

            if validate_scene(scene):

                # narration limiter
                scene["narration"] = scene["narration"][:700]

                valid_scenes.append(scene)

        if len(valid_scenes) == 0:
            return fallback_storyboard(final_answer)

        return valid_scenes

    except Exception as e:

        print("\n[Storyboard Generation Failed]")
        print(e)

        return fallback_storyboard(final_answer)


# =========================================================
# DIAGRAM ENGINE
# =========================================================

def draw_centered_text(draw, text, y, font):

    bbox = draw.textbbox((0, 0), text, font=font)

    width = bbox[2] - bbox[0]

    draw.text(
        ((WIDTH - width) / 2, y),
        text,
        fill="black",
        font=font
    )


def draw_diagram(draw, scene):

    diagram_type = scene["diagram_type"]

    body_font = get_font(28)

    # =====================================================
    # EQUATION
    # =====================================================

    if diagram_type == "equation":

        eq = scene.get(
            "equation_text",
            "E = mc²"
        )

        draw.rectangle(
            [180, 250, 1100, 470],
            outline="black",
            width=4
        )

        eq_font = get_font(52)

        draw_centered_text(
            draw,
            eq,
            340,
            eq_font
        )

    # =====================================================
    # PROCESS FLOW
    # =====================================================

    elif diagram_type == "process_flow":

        items = scene.get("items", [])

        if not items:
            items = ["Input", "Process", "Output"]

        start_x = 100
        y = 320

        box_w = 180
        box_h = 80
        gap = 40

        for idx, item in enumerate(items):

            x = start_x + idx * (box_w + gap)

            draw.rectangle(
                [x, y, x + box_w, y + box_h],
                outline="black",
                width=3
            )

            wrapped = textwrap.fill(item, width=15)

            draw.text(
                (x + 20, y + 20),
                wrapped,
                fill="black",
                font=body_font
            )

            if idx < len(items) - 1:

                draw.line(
                    [
                        x + box_w,
                        y + 40,
                        x + box_w + gap,
                        y + 40
                    ],
                    fill="black",
                    width=3
                )

    # =====================================================
    # COMPARISON
    # =====================================================

    elif diagram_type == "comparison":

        left_label = scene.get("left_label", "A")
        right_label = scene.get("right_label", "B")

        draw.rectangle(
            [100, 180, 520, 540],
            outline="black",
            width=4
        )

        draw.rectangle(
            [760, 180, 1180, 540],
            outline="black",
            width=4
        )

        title_font = get_font(34)

        draw_centered_text(draw, left_label, 200, title_font)
        draw_centered_text(draw, right_label, 200, title_font)

        for i, item in enumerate(scene.get("left_items", [])):

            draw.text(
                (130, 270 + i * 45),
                f"• {item}",
                fill="black",
                font=body_font
            )

        for i, item in enumerate(scene.get("right_items", [])):

            draw.text(
                (790, 270 + i * 45),
                f"• {item}",
                fill="black",
                font=body_font
            )

    # =====================================================
    # TIMELINE
    # =====================================================

    elif diagram_type == "timeline":

        items = scene.get("items", [])

        y = 360

        draw.line(
            [120, y, 1160, y],
            fill="black",
            width=4
        )

        spacing = 1000 // max(1, len(items))

        for idx, item in enumerate(items):

            x = 150 + idx * spacing

            draw.ellipse(
                [x - 12, y - 12, x + 12, y + 12],
                fill="black"
            )

            wrapped = textwrap.fill(item, width=12)

            draw.text(
                (x - 50, y - 80),
                wrapped,
                fill="black",
                font=body_font
            )

    # =====================================================
    # NETWORK
    # =====================================================

    elif diagram_type == "network":

        labels = scene.get("labels", [])

        center_x = WIDTH // 2
        center_y = HEIGHT // 2

        draw.ellipse(
            [center_x - 60, center_y - 60,
             center_x + 60, center_y + 60],
            outline="black",
            width=4
        )

        draw_centered_text(
            draw,
            "Core",
            center_y - 20,
            body_font
        )

        positions = [
            (250, 180),
            (1000, 180),
            (250, 560),
            (1000, 560)
        ]

        for idx, label in enumerate(labels[:4]):

            x, y = positions[idx]

            draw.rectangle(
                [x, y, x + 180, y + 70],
                outline="black",
                width=3
            )

            draw.text(
                (x + 20, y + 20),
                label,
                fill="black",
                font=body_font
            )

            draw.line(
                [center_x, center_y, x + 90, y + 35],
                fill="black",
                width=3
            )

    # =====================================================
    # TEXT ONLY
    # =====================================================

    else:

        body = textwrap.fill(
            scene["narration"],
            width=45
        )

        draw.multiline_text(
            (160, 240),
            body,
            fill="black",
            font=body_font,
            spacing=10
        )


# =========================================================
# CREATE SLIDE
# =========================================================

def create_slide(scene, slide_path):

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        color="white"
    )

    draw = ImageDraw.Draw(img)

    title_font = get_font(50)
    subtitle_font = get_font(32)

    # =====================================================
    # TITLE
    # =====================================================

    draw_centered_text(
        draw,
        scene["title"],
        50,
        title_font
    )

    # =====================================================
    # DIAGRAM
    # =====================================================

    draw_diagram(draw, scene)

    # =====================================================
    # SUBTITLE
    # =====================================================

    subtitle = textwrap.fill(
        scene["subtitle"],
        width=50
    )

    y_text = HEIGHT - 110

    for line in subtitle.split("\n"):

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=subtitle_font
        )

        width = bbox[2] - bbox[0]

        draw.text(
            ((WIDTH - width) / 2, y_text),
            line,
            fill="black",
            font=subtitle_font
        )

        y_text += 40

    img.save(slide_path)


# =========================================================
# OFFLINE TTS
# =========================================================

def generate_tts(text, output_audio_path):

    import pyttsx3

    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    engine = pyttsx3.init()

    engine.setProperty("rate", 145)
    engine.setProperty("volume", 1.0)

    engine.save_to_file(
        text,
        output_audio_path
    )

    engine.runAndWait()

    # ensure file exists
    timeout = 15
    waited = 0

    while not os.path.exists(output_audio_path):

        time.sleep(0.5)

        waited += 0.5

        if waited > timeout:
            raise RuntimeError(
                f"TTS failed: {output_audio_path}"
            )

    # ensure stable size
    previous = -1
    stable = 0

    while stable < 3:

        size = os.path.getsize(output_audio_path)

        if size == previous and size > 0:
            stable += 1
        else:
            stable = 0

        previous = size

        time.sleep(0.5)


# =========================================================
# SRT SUBTITLES
# =========================================================

def seconds_to_srt(sec):

    hrs = int(sec // 3600)
    mins = int((sec % 3600) // 60)
    secs = int(sec % 60)

    return f"{hrs:02}:{mins:02}:{secs:02},000"


def generate_srt(storyboard):

    lines = []

    current_time = 0

    for idx, scene in enumerate(storyboard, start=1):

        duration = max(
            4,
            len(scene["narration"].split()) // 2
        )

        start = current_time
        end = current_time + duration

        lines.append(
            f"{idx}\n"
            f"{seconds_to_srt(start)} --> {seconds_to_srt(end)}\n"
            f"{scene['subtitle']}\n"
        )

        current_time = end

    srt_path = OUTPUT_DIR / "subtitles.srt"

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return srt_path


# =========================================================
# BUILD VIDEO
# =========================================================

def build_video(storyboard):

    video_clips = []

    for idx, scene in enumerate(storyboard):

        slide_file = SLIDES_DIR / f"slide_{idx}.png"
        audio_file = AUDIO_DIR / f"audio_{idx}.wav"

        print(f"\n[Scene {idx + 1}]")

        print("Generating slide...")
        create_slide(scene, str(slide_file))

        print("Generating narration...")
        generate_tts(
            scene["narration"],
            str(audio_file)
        )

        if not os.path.exists(audio_file):
            raise RuntimeError(
                f"Missing audio: {audio_file}"
            )

        audio_clip = AudioFileClip(str(audio_file))

        image_clip = (
            ImageClip(str(slide_file))
            .set_duration(audio_clip.duration)
            .set_audio(audio_clip)
        )

        video_clips.append(image_clip)

        print(
            f"Duration: {audio_clip.duration:.2f} sec"
        )

    print("\nCombining clips...")

    final_video = concatenate_videoclips(
        video_clips,
        method="compose"
    )

    if final_video.audio is None:
        raise RuntimeError(
            "Final video has no audio."
        )

    print("\nRendering final video...")

    final_video.write_videofile(
        str(VIDEO_OUTPUT),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        temp_audiofile=str(
            OUTPUT_DIR / "temp_audio.aac"
        ),
        remove_temp=True,
        threads=4,
        verbose=True
    )

    final_video.close()

    for clip in video_clips:

        if clip.audio:
            clip.audio.close()

        clip.close()

    print(f"\nDone: {VIDEO_OUTPUT}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # EXAMPLE RAG OUTPUT
    # =====================================================

    final_answer = """
    The central dogma of molecular biology describes how genetic information flows from DNA to RNA to protein.
    DNA replication begins at origins of replication where helicase unwinds the double helix by breaking hydrogen bonds.
    Primase lays down RNA primers, and DNA polymerase III adds complementary nucleotides in the 5-prime to 3-prime direction.
    The leading strand is synthesized continuously while the lagging strand is built in Okazaki fragments.
    DNA ligase seals the fragments, producing two identical daughter DNA molecules via semi-conservative replication.
    Transcription begins when RNA polymerase binds the promoter region and synthesizes a pre-mRNA strand.
    The pre-mRNA undergoes splicing where introns are removed by the spliceosome and exons are joined.
    The mature mRNA exits the nucleus and is translated by ribosomes in the cytoplasm.
    Transfer RNAs carry specific amino acids; their anticodons match mRNA codons in the ribosomal A-site.
    Peptide bonds form between amino acids in the growing polypeptide chain until a stop codon triggers release.
    """

    print("\n=================================================")
    print("GENERATING STORYBOARD")
    print("=================================================\n")

    storyboard = generate_storyboard(final_answer)

    print(f"Generated {len(storyboard)} scenes\n")

    for idx, scene in enumerate(storyboard, start=1):

        print(
            f"Scene {idx}: "
            f"{scene['title']} "
            f"({scene['diagram_type']})"
        )

    print("\n=================================================")
    print("GENERATING SUBTITLES")
    print("=================================================\n")

    generate_srt(storyboard)

    print("\n=================================================")
    print("BUILDING VIDEO")
    print("=================================================\n")

    build_video(storyboard)

    print("\n=================================================")
    print("PIPELINE COMPLETE")
    print("=================================================\n")