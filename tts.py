from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip
from moviepy.video.fx.all import crop
from TTS.api import TTS
from mutagen.wave import WAVE
import whisper_timestamped as whisper
import moviepy.editor as mp
import random
import reddit
import torch
import csv
import os


SAMPLE_VOICE = "input.mp3"
SAMPLE_LANGUAGE = "en"
OUTPUT_TTS = "output.wav"
BACKGROUND_VIDEO = "background.mp4"
COMPLETED_POSTS = "previous.json"

OUTPUT_DIR = "output/"
VIDEO_PATH = OUTPUT_DIR + "result.mp4"


def get_input_string():
    # # get input string for reddit api???
    client = reddit.Client("0Cud6lonrCDVlGT8gtqx8w", "	kL-fcgtubLjpi7sVZzeC2kyS8EBqRQ")

    subreddit = client.Subreddit(
        mode="top", name="amitheasshole", limit=100
    )  # supports 'top', 'new' or 'hot' mode of submissions

    for x in range(100):
        print(f"Post number: {x+1}")
        title = subreddit.title(x)
        inner_text = subreddit.selftext(x)
        author = subreddit.author(x)

        if len(inner_text.split()) <= 250:
            if not os.path.exists(COMPLETED_POSTS):
                with open(COMPLETED_POSTS, "w") as f:
                    writer = csv.writer(f)
                    writer.writerows([["Author", "Title"], [author, title]])

            with open(COMPLETED_POSTS, "r+") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] != author or row[1] != title:
                        writer = csv.writer(f)
                        writer.writerow([author, title])

            input_string = f"{title}\n {inner_text}".lower()
            input_string = input_string.replace("aita", "am i the asshole")

            do_tts(input_string)


def do_tts(input_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=input_text,
        speaker_wav=SAMPLE_VOICE,
        language=SAMPLE_LANGUAGE,
        file_path=OUTPUT_TTS,
    )

    audio = WAVE(OUTPUT_TTS)
    if audio.info.length <= 59.8:  # short must be under 1 minute
        transcribe_tts()
    else:
        print(f"Audio clip too long at: {audio.info.length} seconds")


def transcribe_tts():
    audio = whisper.load_audio(OUTPUT_TTS)
    model = whisper.load_model("base")
    result = whisper.transcribe(model, audio, language="en")

    words = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words.append(word)

    length = result["segments"][(len(result["segments"]) - 1)]["end"]

    vid = extract_video(clip_length=length)
    add_subtitles(random_clip=vid, words=words)


def extract_video(clip_length):
    video_clip = mp.VideoFileClip(BACKGROUND_VIDEO)
    video_duration = video_clip.duration
    clip_length = min(clip_length, video_duration)
    start_time = random.uniform(0, video_duration - clip_length)
    rand_clip = video_clip.subclip(start_time, start_time + clip_length)
    return rand_clip.without_audio()


def add_subtitles(random_clip, words):
    text_clips = []
    current_time = 0
    for index, entry in enumerate(words):
        word = entry["text"]
        try:
            duration = words[index + 1]["start"] - entry["start"]
        except:
            duration = entry["end"] - entry["start"]

        text_clip = TextClip(
            word,
            fontsize=100,
            color="white",
            bg_color="transparent",
            stroke_color="black",
            stroke_width=2,
            size=(random_clip.size[0], random_clip.size[1]),
        )

        text_clip = text_clip.set_start(current_time).set_end(current_time + duration)

        current_time += duration

        text_clips.append(text_clip)

    subtitled_clip = CompositeVideoClip([random_clip] + text_clips)

    target_resolution = (1080, 1920)

    final_clip = subtitled_clip.resize(target_resolution)

    add_audio(final_clip)


def add_audio(final_clip):
    audio_clip = AudioFileClip(OUTPUT_TTS)

    final_video = final_clip.set_audio(audio_clip)

    save_video(final_video)


def save_video(video):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    video.write_videofile(VIDEO_PATH, codec="libx264", audio_codec="aac")


get_input_string()
