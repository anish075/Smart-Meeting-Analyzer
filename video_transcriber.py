import os
import argparse
import tempfile
import pandas as pd
import whisper
import torch
from pyannote.audio import Pipeline
from moviepy import VideoFileClip
from pydub import AudioSegment
import librosa
import soundfile as sf
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from denoiser import enhance_audio_deepfilternet, resample_to_48k


class VideoTranscriber:
    # Class-level cache for models (shared across instances)
    _whisper_models = {}
    _diarization_pipeline = None
    
    def __init__(self, whisper_model="large-v3", device=None):
        """
        Initialize the Video Transcriber with state-of-the-art models.
        Uses caching to avoid reloading models multiple times.
        
        Args:
            whisper_model (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Smart device selection with memory check
        if device is None:
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                print(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
                
                if gpu_memory < 8:  # Less than 8GB
                    print(f"⚠️  GPU has {gpu_memory:.1f}GB memory. Recommending CPU for stability.")
                    print("   You can force GPU with --device cuda, but may run out of memory.")
                    self.device = "cpu"
                else:
                    self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.whisper_model_name = whisper_model
        print(f"Using device: {self.device}")
        
        # Set memory management for CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
            # Set memory allocation strategy
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Load Whisper model with caching
        model_key = f"{whisper_model}_{self.device}"
        if model_key in VideoTranscriber._whisper_models:
            print(f"Using cached Whisper model ({whisper_model})...")
            self.whisper_model = VideoTranscriber._whisper_models[model_key]
        else:
            print(f"Loading Whisper model ({whisper_model}) - this may take a while on first run...")
            self.whisper_model = whisper.load_model(whisper_model, device=self.device)
            VideoTranscriber._whisper_models[model_key] = self.whisper_model
            print("✅ Whisper model loaded and cached for future use!")
        
        # Load diarization pipeline with caching
        if VideoTranscriber._diarization_pipeline is None:
            print("Loading pyannote diarization pipeline - this may take a while on first run...")
            # PASTE YOUR HUGGING FACE TOKEN HERE (replace the text between quotes)
            HUGGINGFACE_TOKEN = "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"
            
            VideoTranscriber._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HUGGINGFACE_TOKEN
            )
            if torch.cuda.is_available():
                VideoTranscriber._diarization_pipeline.to(torch.device("cuda"))
            print("✅ Diarization pipeline loaded and cached for future use!")
        else:
            print("Using cached diarization pipeline...")
        
        self.diarization_pipeline = VideoTranscriber._diarization_pipeline
    
    def extract_audio_from_video(self, video_path, output_audio_path):
        """
        Extract audio from MP4 video file.
        
        Args:
            video_path (str): Path to input video file
            output_audio_path (str): Path for extracted audio file
        """
        print("Extracting audio from video...")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(output_audio_path, fps=48000, nbytes=2, codec="pcm_s16le", logger=None)
        clip.close()
        print("Audio extraction complete.")
    
    def apply_noise_reduction(self, audio_path):
        """
        Apply noise reduction using the existing denoiser functionality.
        
        Args:
            audio_path (str): Path to audio file to denoise
        """
        print("Applying noise reduction...")
        
        resample_to_48k(audio_path)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_enhanced:
            enhanced_audio_path = temp_enhanced.name
        
        enhance_audio_deepfilternet(audio_path, enhanced_audio_path)
        
        os.replace(enhanced_audio_path, audio_path)
        print("Noise reduction complete.")
    
    def perform_diarization(self, audio_path):
        """
        Perform speaker diarization on the audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            pyannote.core.Annotation: Diarization results
        """
        print("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        print(f"Diarization complete. Found {len(diarization.labels())} speakers.")
        return diarization
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using Whisper with word-level timestamps.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Whisper transcription result with word timestamps
        """
        print("Transcribing audio with Whisper...")
        
        # Check if file exists before transcription
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Audio file size: {os.path.getsize(audio_path) / (1024*1024):.1f} MB")
        
        try:
            # Convert to absolute path and normalize
            audio_path = os.path.abspath(audio_path)
            print(f"Using absolute path: {audio_path}")
            
            # Load audio with librosa (bypasses FFmpeg issues)
            print("Loading audio with librosa to bypass FFmpeg...")
            import librosa
            import numpy as np
            
            # Load the entire audio file with librosa
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
            print(f"✅ Audio loaded: {len(audio_data)/sample_rate:.1f} seconds at {sample_rate}Hz")
            
            # Pass numpy array directly to Whisper (no file path needed)
            result = self.whisper_model.transcribe(
                audio_data,  # Pass numpy array instead of file path
                word_timestamps=True,
                language=None,  # Auto-detect language
                verbose=False
            )
            print("Transcription complete.")
            return result
            
        except Exception as e:
            print(f"Whisper transcription error: {str(e)}")
            print(f"Audio path: {audio_path}")
            print(f"Path exists: {os.path.exists(audio_path)}")
            print(f"Path is file: {os.path.isfile(audio_path)}")
            
            # If numpy array approach fails, try the old file path method
            try:
                print("Trying direct file path method as fallback...")
                result = self.whisper_model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    language=None,
                    verbose=False
                )
                print("Transcription complete (using file path).")
                return result
                
            except Exception as e2:
                print(f"File path approach also failed: {str(e2)}")
                print("\n🚨 SOLUTION: Install FFmpeg to fix this issue:")
                print("1. Download FFmpeg from: https://ffmpeg.org/download.html")
                print("2. Add FFmpeg to your Windows PATH")
                print("3. Or use: winget install FFmpeg")
                raise e
    
    def format_timestamp(self, seconds):
        """
        Convert seconds to HH:MM:SS format.
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def merge_diarization_transcription(self, diarization, transcription):
        """
        Merge speaker diarization with transcription results.
        
        Args:
            diarization: pyannote diarization results
            transcription: Whisper transcription results
            
        Returns:
            list: List of dictionaries with timestamp, speaker, and transcript
        """
        print("Merging diarization and transcription results...")
        
        results = []
        
        for segment in transcription["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_text = segment["text"].strip()
            
            if not segment_text:
                continue
            
            segment_middle = (segment_start + segment_end) / 2
            
            speaker_label = "Speaker 1"  # Default
            for turn, track, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= segment_middle <= turn.end:
                    speaker_num = list(diarization.labels()).index(speaker) + 1
                    speaker_label = f"Speaker {speaker_num}"
                    break
            
            results.append({
                "Timestamp": self.format_timestamp(segment_start),
                "Speaker": speaker_label,
                "Transcript": segment_text
            })
        
        if not results:
            for segment in transcription["segments"]:
                results.append({
                    "Timestamp": self.format_timestamp(segment["start"]),
                    "Speaker": "Speaker 1",
                    "Transcript": segment["text"].strip()
                })
        
        print(f"Merged {len(results)} transcript segments with speaker information.")
        return results
    
    def save_to_excel(self, results, output_path):
        """
        Save transcription results to Excel file.
        
        Args:
            results (list): List of transcription results
            output_path (str): Path for output Excel file
        """
        print("Saving results to Excel...")
        
        df = pd.DataFrame(results)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Transcription', index=False)
            
            worksheet = writer.sheets['Transcription']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Excel file saved: {output_path}")
        print(f"Total segments: {len(results)}")
    
    def process_video(self, input_video_path, output_excel_path, apply_denoising=True):
        """
        Complete pipeline: video to Excel transcription with speaker diarization.
        
        Args:
            input_video_path (str): Path to input MP4 video
            output_excel_path (str): Path for output Excel file
            apply_denoising (bool): Whether to apply noise reduction
        """
        print(f"Processing video: {input_video_path}")
        print(f"Output will be saved to: {output_excel_path}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            self.extract_audio_from_video(input_video_path, temp_audio_path)
            print(f"✅ Audio extracted to: {temp_audio_path}")
            
            if apply_denoising:
                self.apply_noise_reduction(temp_audio_path)
                print(f"✅ Audio after denoising: {os.path.getsize(temp_audio_path) / (1024*1024):.1f} MB")
            
            print(f"🔍 Audio file exists: {os.path.exists(temp_audio_path)}")
            diarization = self.perform_diarization(temp_audio_path)
            
            print(f"🔍 Audio file still exists before transcription: {os.path.exists(temp_audio_path)}")
            transcription = self.transcribe_audio(temp_audio_path)
            
            results = self.merge_diarization_transcription(diarization, transcription)
            
            self.save_to_excel(results, output_excel_path)
            
            print("\n" + "="*50)
            print("TRANSCRIPTION COMPLETE!")
            print("="*50)
            print(f"Input video: {input_video_path}")
            print(f"Output Excel: {output_excel_path}")
            print(f"Total speakers detected: {len(diarization.labels())}")
            print(f"Total transcript segments: {len(results)}")
            print("="*50)
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                print("Temporary audio file cleaned up.")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Video Transcriber with Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python video_transcriber.py input.mp4 output.xlsx
    python video_transcriber.py lecture.mp4 transcript.xlsx --no-denoise
    python video_transcriber.py meeting.mp4 result.xlsx --model large-v3
        """
    )
    
    parser.add_argument(
        "input_video",
        help="Path to input MP4 video file"
    )
    
    parser.add_argument(
        "output_excel", 
        help="Path for output Excel file"
    )
    
    parser.add_argument(
        "--model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
        help="Whisper model size (default: large-v3 for best accuracy)"
    )
    
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip noise reduction step"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for processing (default: auto)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return 1
    
    device = None if args.device == "auto" else args.device
    apply_denoising = not args.no_denoise
    
    print("Initializing Advanced Video Transcriber...")
    print(f"Model: {args.model}")
    print(f"Noise reduction: {'Enabled' if apply_denoising else 'Disabled'}")
    
    try:
        transcriber = VideoTranscriber(whisper_model=args.model, device=device)
        transcriber.process_video(args.input_video, args.output_excel, apply_denoising)
        return 0
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())