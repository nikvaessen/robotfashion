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
    def __init__(self, path: pathlib.Path, queue: Queue, ctx: rs.context, dev_id: str):
        super().__init__()

        self.path: pathlib.Path = path
        self.queue: Queue = queue
        self.ctx = ctx
        self.dev_id: str = dev_id

        self._mark_event = Event()
        self._frame_count = 0
        self._start_ts = time()

    def run(self) -> None:
        # pipes = []

        # for dev in self.rsctx.query_devices():
        pipe = rs.pipeline(self.ctx)
        # pipes.append(pipe)

        config = rs.config()
        config.enable_device(self.dev_id)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, DESIRED_FPS)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, DESIRED_FPS)

        pipe.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        for i in range(0, DESIRED_FPS * 3):
            # for idx, pipe in enumerate(pipes):
            pipe.wait_for_frames()

        try:
            self._frame_count = 0
            self._start_ts = time()

            while not self.stopped():
                self._frame_count += 1
                try:
                    # depth_frames = []
                    # color_frames = []

                    # for pipe in pipes:
                    frames: rs.composite_frame = pipe.wait_for_frames()
                    aligned_frames = align.process(frames)

                    cf: rs.video_frame = aligned_frames.get_color_frame()
                    df: rs.depth_frame = aligned_frames.get_depth_frame()

                    # color_frames.append(color_frame)
                    # depth_frames.append(depth_frame)

                    # for dev_idx, (cf, df) in enumerate(zip(color_frames, depth_frames)):
                    color_image = np.asanyarray(cf.get_data())
                    depth_image = np.asanyarray(cf.get_data())

                    color_package = FramePackage(
                        color_image,
                        self.path
                        / "dev_{}_frame_{}_color".format(
                            self.dev_id, self._frame_count
                        ),
                        self.is_marked(),
                    )

                    depth_package = FramePackage(
                        depth_image,
                        self.path
                        / "dev_{}_frame_{}_depth".format(
                            self.dev_id, self._frame_count
                        ),
                        self.is_marked(),
                    )

                    self.queue.put(color_package, block=False)
                    self.queue.put(depth_package, block=False)

                    self._reset_mark()
                except RuntimeError:
                    pass
        finally:
            # for pipe in pipes:
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
                self.queue.task_done()
            except Empty:
                pass

    @staticmethod
    def _save_package(package: FramePackage):
        np.savez_compressed(
            package.storage_path, data=package.data, marked=package.marked
        )

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

        self.recording_threads = []
        self.image_processing_threads = []

        self.data_queue = Queue()

        for n in range(0, 5):
            image_processing_thread: ImageProcessingThread = ImageProcessingThread(
                self.data_queue
            )
            image_processing_thread.start()
            self.image_processing_threads.append(image_processing_thread)

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
                        for pt in self.image_processing_threads:
                            pt.stop()
                            pt.join()
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
                for rt in self.recording_threads:
                    rt.mark_next_frame()

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
            recording_thread = self.recording_threads[0]

            recording_time = recording_thread.get_recording_time()
            frame_count = recording_thread.get_frame_count()
            fps = recording_thread.get_estimated_fps()

            self.win.addstr(
                "\nrecording time: {:5.0f} seconds\t".format(recording_time)
            )
            self.win.addstr("processed frames: {}\t".format(frame_count))
            self.win.addstr("estimated FPS: {:5.1f}\t".format(fps))
            self.win.addstr("queue size: {}".format(self.data_queue.qsize()))

            self.win.addstr(
                winH - 1, 0, 'press "m" to mark a frame, "s" to stop recording'
            )

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

        device_ids = []

        for dev in self.rsctx.query_devices():
            device_ids.append(dev.get_info(rs.camera_info.serial_number))

        self.recording_threads.clear()

        for dev_id in device_ids:
            record_thread = RecordingThread(
                pathlib.Path(self.current_dir), self.data_queue, self.rsctx, dev_id
            )
            record_thread.start()
            self.recording_threads.append(record_thread)

    def stop_recording(self):
        for rt in self.recording_threads:
            rt.stop()

        for rt in self.recording_threads:
            rt.join()

        self.data_queue.join()

        self.state = State.MAIN

    def display_error(self, message: str, display_time_ms: float):
        self.error = message
        self.error_deadline = time() + display_time_ms / 1000


def main(win):
    rsctx = rs.context()

    manager = UiManager(win, rsctx)
    manager.start()


def record_without_gui_to_see_exceptions():
    ctx = rs.context()

    device_ids = []

    for dev in ctx.query_devices():
        device_ids.append(dev.get_info(rs.camera_info.serial_number))

    print("recognized {} devices".format(len(device_ids)))

    queue = Queue()
    processing_thread = ImageProcessingThread(queue)
    processing_thread.start()

    recording_threads = []
    for dev_id in device_ids:
        record_thread = RecordingThread(pathlib.Path(os.getcwd()), queue, ctx, dev_id)
        record_thread.start()
        recording_threads.append(record_thread)

    print("starting sleep")
    sleep(10)
    print("ending sleep")

    for rt in recording_threads:
        print(rt.get_estimated_fps(), rt.get_frame_count(), rt.get_recording_time())
        rt.stop()

    for rt in recording_threads:
        rt.join()

    queue.join()

    processing_thread.stop()
    processing_thread.join()

    for file in os.listdir(os.getcwd()):
        f = np.load(file)
        data = f["data"]
        marked = f["marked"]

        if "color" in file:
            img_fn = file.split(".")[0] + ".png"
            imageio.imwrite(img_fn, data)


def curses_no_render():
    rsctx = rs.context()

    manager = UiManager(None, rsctx)
    manager.start_recording()

    sleep(10)

    manager.stop_recording()


if __name__ == "__main__":
    curses.wrapper(main)
    # record_without_gui_to_see_exceptions()
    # curses_no_render()
