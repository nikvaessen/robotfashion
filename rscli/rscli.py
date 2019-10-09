import curses
import pathlib
import os

from time import sleep, time
from enum import Enum
from threading import Thread, Event
from typing import Callable, List
from queue import Queue, Empty

try:
    import pyrealsense2 as rs
    import numpy as np
    import imageio
except ImportError:
    rs = None
    print("One or more dependencies not found. Run 'pip install -r requirements.txt'")
    exit(1)


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
    def __init__(self, path: pathlib.Path, queue: Queue):
        super().__init__()

        self.path: pathlib.Path = path
        self.queue = queue

        self._mark_event = Event()
        self._frame_count = 0
        self._start_ts = time()

    def run(self) -> None:
        pipe = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        profile = pipe.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        for i in range(0, 18):
            pipe.wait_for_frames()

        try:
            self._frame_count = 0
            self._start_ts = time()

            while not self.stopped():
                self._frame_count += 1

                try:
                    frames: rs.composite_frame = pipe.wait_for_frames()
                    aligned_frames = align.process(frames)

                    color_frame: rs.video_frame = aligned_frames.get_color_frame()
                    depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()

                    color_frame.keep()

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())

                    color_package = FramePackage(
                        color_image,
                        self.path / "{}_color".format(self._frame_count),
                        self.is_marked(),
                    )

                    depth_package = FramePackage(
                        depth_image,
                        self.path / "{}_depth".format(self._frame_count),
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
        return time() - self._start_ts

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
            except Empty:
                pass

    @staticmethod
    def _save_package(package: FramePackage):
        np.savez_compressed(package.storage_path, data=package.data)

        if package.marked:
            imageio.imwrite(str(package.storage_path) + ".png", package.data)


class State(Enum):
    MAIN = 0
    RECORDING = 1
    ASKING_WRITTEN_INPUT = 2


class UiManager:
    def __init__(self, window, rsctx: rs.context):
        self.state: State = State.MAIN
        self.devices_connected: int = 0
        self.win = window
        self.rsctx: rs.context = rsctx
        self.current_dir: pathlib.Path = pathlib.Path(os.getcwd())

        self.written_input_ready: Callable[[str], None] = None
        self.written_input: str = None
        self.written_input_query: str = None

        self.error: str = None
        self.error_deadline: float = None

        self.recording_thread: RecordingThread = None

        self.data_queue = Queue(100)
        self.image_processing_thread: ImageProcessingThread = ImageProcessingThread(
            self.data_queue
        )
        self.image_processing_thread.start()

    def handle_keyboard_input(self):
        key = self._get_input_key()

        if key is None:
            return

        if self.state == State.ASKING_WRITTEN_INPUT:
            if key == "\n":
                self.state = State.MAIN
                self.written_input_ready(self.written_input)
                self.written_input = ""
            elif key == "KEY_BACKSPACE":
                self.written_input = self.written_input[0:-1]
            else:
                self.written_input += key

        elif self.state == State.MAIN:
            if key == "f":
                self.state = State.ASKING_WRITTEN_INPUT

                def f(written):
                    path = pathlib.Path(written)

                    if not path.exists():
                        path.mkdir(parents=True)

                    self.current_dir = path

                self.written_input = ""
                self.written_input_ready = f
                self.written_input_query = "New storage directory path: "
                self.written_input = str(self.current_dir)

            elif key == "q":
                self.state = State.ASKING_WRITTEN_INPUT

                def f(written):
                    if written == "y" or written == "yes":
                        self.image_processing_thread.stop()
                        self.image_processing_thread.join()
                        exit(0)

                self.written_input = ""
                self.written_input_ready = f
                self.written_input_query = "Are you sure you want to quit? [y\\n]"

            elif key == "r":
                self.start_recording()

        elif self.state == State.RECORDING:
            if key == "s":
                self.stop_recording()

            elif key == "m":
                self.recording_thread.mark_next_frame()

    def _get_input_key(self) -> str or None:
        try:
            return self.win.getkey()
        except curses.error as e:
            return None

    def _update_num_devices(self):
        devices = self.rsctx.query_devices()
        self.devices_connected = len(devices)

    def update(self):
        self._update_num_devices()

    def render(self):
        winH, winW = self.win.getmaxyx()

        self.win.clear()

        self.win.addstr("connected devices: {}\t".format(self.devices_connected))
        self.win.addstr("recording: {}\n".format(self.state == State.RECORDING))
        self.win.addstr("\nstorage directory: {}\n".format(self.current_dir))

        if self.error_deadline is not None:
            self.win.addstr("\n\n\n")
            self.win.addstr("error: {}".format(self.error))

            if time() > self.error_deadline:
                self.error = ""
                self.error_deadline = None

        if self.state == State.MAIN:
            self.win.addstr(
                winH - 1,
                0,
                'press "r" to start recording, "f" to change storage folder, "q" to quit',
            )
        elif self.state == State.ASKING_WRITTEN_INPUT:
            self.win.addstr(
                winH - 1, 0, self.written_input_query + " " + self.written_input
            )

        elif self.state == State.RECORDING:
            recording_time = self.recording_thread.get_recording_time()
            frame_count = self.recording_thread.get_frame_count()
            fps = self.recording_thread.get_estimated_fps()

            self.win.addstr(
                "\nrecording time: {:5.0f} seconds\t".format(recording_time)
            )
            self.win.addstr("processed frames: {}\t".format(frame_count))
            self.win.addstr("estimated FPS: {:5.1f}\t".format(fps))
            self.win.addstr("queue size: {}".format(self.data_queue.qsize()))

            self.win.addstr(winH - 1, 0, 'press "m" to mark a frame, "s" to stop recording')

    def start(self):
        self.win.nodelay(True)

        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)
        curses.color_pair(1)

        prev_frame_update_ts = time()
        prev_device_update_ts = time()

        while True:
            sleep(1 / 1000)
            current_frame_ts = time()

            self.handle_keyboard_input()

            if current_frame_ts - prev_frame_update_ts > 1 / 15:
                self.render()
                prev_frame_update_ts = current_frame_ts

            if current_frame_ts - prev_device_update_ts > 1:
                self.update()
                prev_device_update_ts = current_frame_ts

    def start_recording(self):
        self.update()

        if self.devices_connected <= 0:
            self.display_error("no camera(s) connected", 1000)
            return

        self.state = State.RECORDING

        self.recording_thread = RecordingThread(self.current_dir, self.data_queue)
        self.recording_thread.start()

    def stop_recording(self):
        self.recording_thread.stop()
        self.recording_thread.join()

        self.data_queue.join()

        self.state = State.MAIN

    def display_error(self, message: str, display_time_ms: float):
        self.error = message
        self.error_deadline = time() + display_time_ms / 1000


def main(win):
    rsctx = rs.context()

    manager = UiManager(win, rsctx)
    manager.start()


if __name__ == "__main__":
    curses.wrapper(main)
