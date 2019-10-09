import curses
import pathlib
import os

from time import sleep, time
from enum import Enum


try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("pyrealsense2 library not found. Did you run `pip install pyrealsense2'?")
    exit(1)


class State(Enum):
    MAIN = 0
    RECORDING = 1
    ASKING_WRITTEN_INPUT = 2


class UiManager:
    def __init__(self, window, rsctx):
        self.state = State.MAIN
        self.devices_connected = 0
        self.win = window
        self.rsctx = rsctx
        self.current_dir = pathlib.Path(os.getcwd())

        self.written_input_ready = None
        self.written_input = ""
        self.written_input_query = ""

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

                self.written_input_ready = f
                self.written_input_query = "New storage directory path: "
                self.written_input = str(self.current_dir)

            elif key == "q":
                self.state = State.ASKING_WRITTEN_INPUT

                def f(written):
                    if written == "y" or written == "yes":
                        exit(0)

                self.written_input_ready = f
                self.written_input_query = "Are you sure you want to quit? [y\\n]"

            elif key == "r":
                self.start_recording()

        elif self.state == State.RECORDING:
            if key == "s":
                self.stop_recording()

    def _get_input_key(self):
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
        self.win.addstr("\nstorage directory: {}".format(self.current_dir))

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
            self.win.addstr(winH - 1, 0, 'press "s" to stop recording')

            self.start_recording()

    def start(self):
        self.win.nodelay(True)

        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)
        curses.color_pair(1)

        prev_frame_update = time()

        while True:
            sleep(1 / 1000)
            current_frame_ts = time()

            if current_frame_ts - prev_frame_update > 1 / 30:
                self.update()
                self.render()
                self.handle_keyboard_input()

                prev_frame_update = current_frame_ts

    def start_recording(self):
        self.state = State.RECORDING
        pass

    def stop_recording(self):
        self.state = State.MAIN

        pass


def main(win):
    rsctx = rs.context()

    manager = UiManager(win, rsctx)
    manager.start()


if __name__ == "__main__":
    curses.wrapper(main)
