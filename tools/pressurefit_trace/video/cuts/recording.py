"""Stream held/fading frames while recording their exact replay states."""


class Recording:
    def __init__(self, renderer, stream, fps):
        self.renderer = renderer
        self.stream = stream
        self.fps = fps
        self.frame_count = 0
        self.shots = []

    def emit(self, seconds=0, *, flash=False, frames=None):
        frames = max(1, round(seconds * self.fps)) if frames is None else frames
        self.shots.append(
            {
                **self.renderer.last_record,
                "first_video_frame": self.frame_count,
                "video_frame_count": frames,
            }
        )
        if flash:
            fade_frames = min(frames, max(2, round(0.8 * self.fps)))
            for i in range(fade_frames):
                self.stream.write(
                    self.renderer.rgba(flash=(1 - i / max(1, fade_frames - 1)) ** 2)
                )
            rgba = self.renderer.rgba(flash=0)
            for _ in range(frames - fade_frames):
                self.stream.write(rgba)
        else:
            rgba = self.renderer.rgba()
            for _ in range(frames):
                self.stream.write(rgba)
        self.frame_count += frames
