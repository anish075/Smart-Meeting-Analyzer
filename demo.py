#!/usr/bin/env python3
"""
Demo script for Advanced Video Transcriber
Demonstrates basic usage and expected output format.
"""

import os
import sys
from video_transcriber import VideoTranscriber

def demo_transcription():
    """
    Demo function showing how to use the VideoTranscriber class.
    """
    print("🎬 Advanced Video Transcriber Demo")
    print("=" * 50)
    
    # Check if a demo video file is provided
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_video_file>")
        print("\nExample:")
        print("python demo.py sample_meeting.mp4")
        print("\nThis will create 'demo_transcript.xlsx' with the results.")
        return
    
    input_video = sys.argv[1]
    output_excel = "demo_transcript.xlsx"
    
    # Verify input file exists
    if not os.path.exists(input_video):
        print(f"❌ Error: Video file not found: {input_video}")
        return
    
    print(f"📹 Input video: {input_video}")
    print(f"📊 Output Excel: {output_excel}")
    print()
    
    try:
        # Initialize transcriber with medium model for faster demo
        print("🚀 Initializing transcriber (using 'medium' model for demo)...")
        transcriber = VideoTranscriber(whisper_model="medium")
        
        # Process the video
        transcriber.process_video(input_video, output_excel, apply_denoising=True)
        
        # Show success message
        print(f"\n✅ Demo complete! Check '{output_excel}' for results.")
        print("\n📋 Expected Excel format:")
        print("┌─────────────┬───────────┬─────────────────────────────┐")
        print("│ Timestamp   │ Speaker   │ Transcript                  │")
        print("├─────────────┼───────────┼─────────────────────────────┤")
        print("│ 00:00:01    │ Speaker 1 │ Hello, welcome everyone.    │")
        print("│ 00:00:04    │ Speaker 2 │ Thank you for having us.    │")
        print("│ 00:00:07    │ Speaker 1 │ Let's begin with the agenda │")
        print("└─────────────┴───────────┴─────────────────────────────┘")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. For pyannote.audio, you need to login: huggingface-cli login")
        print("3. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("4. Ensure you have sufficient RAM and disk space")

def show_system_info():
    """Show system information for debugging."""
    print("\n🔍 System Information:")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import whisper
        print(f"Whisper: Available")
    except ImportError:
        print("Whisper: Not installed")
    
    try:
        from pyannote.audio import Pipeline
        print(f"pyannote.audio: Available")
    except ImportError:
        print("pyannote.audio: Not installed")
    
    try:
        import pandas
        print(f"Pandas version: {pandas.__version__}")
    except ImportError:
        print("Pandas: Not installed")

if __name__ == "__main__":
    # Show system info if requested
    if len(sys.argv) > 1 and sys.argv[1] in ["--info", "-i", "info"]:
        show_system_info()
    else:
        demo_transcription()