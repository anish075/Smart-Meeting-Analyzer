import os
import argparse
import tempfile
import librosa
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip
from df.enhance import enhance, init_df, load_audio, save_audio


def resample_to_48k(audio_path):
    print("Resampling audio to 48kHz if needed...")
    audio, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != 48000:
        audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=48000)
        sf.write(audio_path, audio_resampled, 48000)
        print(f"Audio resampled from {orig_sr} Hz to 48000 Hz.")
    else:
        print("Audio is already at 48kHz.")


def enhance_audio_deepfilternet(input_audio_path, output_audio_path):
    print("Initializing DeepFilterNet model...")
    model, df_state, _ = init_df()

    print("Loading audio for enhancement...")
    audio, _ = load_audio(input_audio_path, sr=df_state.sr())

    print("Enhancing audio with DeepFilterNet...")
    enhanced = enhance(model, df_state, audio)

    print("Saving enhanced audio...")
    save_audio(output_audio_path, enhanced, df_state.sr())
    print("Audio enhancement complete.")


def process_video(input_video_path, output_video_path):
    print("Loading video...")
    clip = VideoFileClip(input_video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    print("Extracting audio from video...")
    clip.audio.write_audiofile(temp_audio_path, fps=48000, nbytes=2, codec="pcm_s16le")

    resample_to_48k(temp_audio_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_enhanced_file:
        temp_enhanced_audio_path = temp_enhanced_file.name
    enhance_audio_deepfilternet(temp_audio_path, temp_enhanced_audio_path)

    print("Attaching enhanced audio to video...")
    new_audio_clip = AudioFileClip(temp_enhanced_audio_path)
    new_clip = clip.with_audio(new_audio_clip)

    print("Writing final video file...")
    new_clip.write_videofile(output_video_path, audio_codec="aac")

    os.remove(temp_audio_path)
    os.remove(temp_enhanced_audio_path)
    print("Temporary files removed. Process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Audio Denoiser")
    parser.add_argument(
        "input_video", help="Path to the input video file", default="input_video.mp4"
    )
    parser.add_argument(
        "output_video",
        help="Path for the output video file",
        default="output_video.mp4",
    )
    args = parser.parse_args()

    process_video(args.input_video, args.output_video)
