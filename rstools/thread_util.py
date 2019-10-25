import pathlib
import time

from queue import Queue, Empty
from threading import Thread, Event

import numpy as np
import pyrealsense2 as rs


class StoppableThread(Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class FramePackage:
    def __init__(self, data: np.ndarray, path: pathlib.Path, marked: bool):
        self.data: np.ndarray = data
        self.storage_path: pathlib.Path = path
        self.marked = marked


class RecordingThread(StoppableThread):
    def __init__(
        self,
        path: pathlib.Path,
        queue: Queue,
        ctx: rs.context,
        dev_id: str,
        fps=6,
        filename_id="",
    ):
        super().__init__()

        self.path: pathlib.Path = path
        self.queue: Queue = queue
        self.ctx = ctx
        self.dev_id: str = dev_id

        self._mark_event = Event()
        self._frame_count = 0
        self._start_ts = time.time()

        self.fps = fps

        self.filename_id = filename_id

    def run(self) -> None:
        pipe = rs.pipeline(self.ctx)

        config = rs.config()
        config.enable_device(self.dev_id)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, self.fps)

        pipe.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        for i in range(0, self.fps * 3):
            pipe.wait_for_frames()

        try:
            self._frame_count = 0
            self._start_ts = time.time()

            while not self.stopped():
                try:

                    frames: rs.composite_frame = pipe.wait_for_frames()
                    aligned_frames = align.process(frames)

                    self._frame_count += 1

                    cf: rs.video_frame = aligned_frames.get_color_frame()
                    df: rs.depth_frame = aligned_frames.get_depth_frame()

                    color_image = np.asanyarray(cf.get_data())
                    depth_image = np.asanyarray(df.get_data())

                    color_package = FramePackage(
                        color_image,
                        self.path
                        / "{}dev_{}_frame_{}_color".format(
                            self.filename_id, self.dev_id, self._frame_count
                        ),
                        self.is_marked(),
                    )

                    depth_package = FramePackage(
                        depth_image,
                        self.path
                        / "{}dev_{}_frame_{}_depth".format(
                            self.filename_id, self.dev_id, self._frame_count
                        ),
                        self.is_marked(),
                    )

                    self.queue.put(color_package, block=False)
                    self.queue.put(depth_package, block=False)

                    self._reset_mark()
                except RuntimeError:
                    pass
        finally:
            pipe.stop()

    def get_frame_count(self):
        return self._frame_count

    def get_recording_time(self):
        return time.time() - self._start_ts

    def get_estimated_fps(self):
        fc = self.get_frame_count()
        rt = self.get_recording_time()

        if fc <= 0:
            return 0
        else:
            return fc / rt

    def mark_next_frame(self):
        self._mark_event.set()

    def is_marked(self):
        return self._mark_event.isSet()

    def _reset_mark(self):
        self._mark_event.clear()


class ImageProcessingThread(StoppableThread):
    def __init__(self, queue: Queue):
        super().__init__()

        self.queue: Queue = queue

    def run(self) -> None:
        while not self.stopped():
            try:
                package = self.queue.get(timeout=1)
                self._save_package(package)
                self.queue.task_done()
            except Empty:
                pass

    @staticmethod
    def _save_package(package: FramePackage):
        np.savez_compressed(
            package.storage_path, data=package.data, marked=package.marked
        )
