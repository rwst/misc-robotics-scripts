"""Video encoding utilities using FFmpeg."""
import subprocess
import sys
from pathlib import Path


class FFmpegEncoder:
    """Streams frames directly to FFmpeg for multi-threaded encoding."""

    def __init__(self, output_path, width, height, fps=30, threads=4):
        """
        Initialize the FFmpeg encoder.

        Args:
            output_path: Path to output video file
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second for output video
            threads: Number of encoding threads
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-threads', str(threads),
            '-c:v', 'libopenh264',
            '-pix_fmt', 'yuv420p',
            str(self.output_path)
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.frame_count = 0
        self._closed = False

    def add_frame(self, frame):
        """
        Add a frame to the video.

        Args:
            frame: numpy array of shape (H, W, 3) with RGB values
        """
        if self._closed:
            return

        # Ensure frame is contiguous and correct type
        import numpy as np
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        try:
            self.proc.stdin.write(frame.tobytes())
            self.frame_count += 1
        except BrokenPipeError:
            self._closed = True
            # Read stderr to get error message
            stderr = self.proc.stderr.read().decode('utf-8', errors='replace')
            print(f"FFmpeg error: {stderr}", file=sys.stderr)
            raise RuntimeError(f"FFmpeg process died. Error: {stderr}")

    def close(self):
        """
        Finalize video encoding.

        Returns:
            int: Number of frames encoded
        """
        if self._closed:
            return self.frame_count

        self._closed = True
        self.proc.stdin.close()
        self.proc.wait()

        if self.proc.returncode != 0:
            stderr = self.proc.stderr.read().decode('utf-8', errors='replace')
            print(f"FFmpeg warning (exit code {self.proc.returncode}): {stderr}", file=sys.stderr)

        return self.frame_count
