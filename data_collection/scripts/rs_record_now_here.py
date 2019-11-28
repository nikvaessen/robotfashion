#! /usr/bin/env python3

import os
import pathlib
import time

from queue import Queue
from threading import Thread

import rstools

import numpy as np
import pyrealsense2 as rs

# LYCON LIBRARY COLLIDES WITH RSTOOLS SO IMPORT IT LAST
import lycon


class PngConverterThread(Thread):
    def __init__(self, queue: Queue):
        super().__init__()
        self.q: Queue = queue

    def run(self) -> None:
        while not self.q.empty():
            pass
            filename, data = self.q.get()
            lycon.save(filename, data)
            self.q.task_done()


class Manager:
    def __init__(self, dir_path: pathlib.Path, num_processing_threads=5):
        self.ctx: rs.context = rs.context()
        self.dir_path: pathlib = dir_path

        self.device_ids = []
        self.processing_threads = []
        self.queue = Queue()
        self.recording_threads = []

        self.frame_delta = []

        self.num_processing_threads = num_processing_threads

    def start_recording(self):
        self.device_ids.clear()
        self.device_ids += rstools.query_device_ids(self.ctx)

        print("recognized {} devices: {}".format(len(self.device_ids), self.device_ids))

        filename_id = str(int(time.time())) + "_"

        for _ in range(0, self.num_processing_threads):
            processing_thread = rstools.ImageProcessingThread(self.queue)
            processing_thread.start()

            self.processing_threads.append(processing_thread)

        for dev_id in self.device_ids:
            record_thread = rstools.RecordingThread(
                pathlib.Path(self.dir_path),
                self.queue,
                self.ctx,
                dev_id,
                filename_id=filename_id,
            )
            record_thread.start()

            self.recording_threads.append(record_thread)

    def stop_threads(self):
        for rt in self.recording_threads:
            print(rt.get_estimated_fps(), rt.get_frame_count(), rt.get_recording_time())
            rt.stop()

        for rt in self.recording_threads:
            rt.join()

        self.queue.join()

        for pt in self.processing_threads:
            pt.stop()

        for pt in self.processing_threads:
            pt.join()

    def print_state(self):
        if len(self.frame_delta) == 0:
            self.frame_delta = [0 for _ in self.recording_threads]

        for idx, rt in enumerate(self.recording_threads):
            fps = rt.get_estimated_fps()
            frames = rt.get_frame_count()
            ts = rt.get_recording_time()
            delta = frames - self.frame_delta[idx]

            self.frame_delta[idx] = frames

            print(
                "device {}: fps={:.3f}, delta={} total frames={} recording time={:.3f}".format(
                    idx, fps, delta, frames, ts
                )
            )

        print("\n")

    def convert_image(self):
        queue = Queue()

        for file in os.listdir(str(self.dir_path)):
            if "npz" in file and "color" in file:
                np_file = np.load(os.path.join(str(self.dir_path), file))
                data = np_file["data"]

                filename = file.split(".")[0] + ".png"
                filename = os.path.join(str(self.dir_path), "pictures", filename)
                print(filename, data.shape)

                queue.put((filename, data))

        threads = []
        for _ in range(self.num_processing_threads):
            t = PngConverterThread(queue)
            t.start()

            threads.append(t)

        for t in threads:
            t.join()


def main():
    path = os.getcwd()

    actual_path = os.path.join(path, time.strftime("%Y-%m-%d_%H-%M-%S"))
    picture_path = os.path.join(actual_path, "pictures")
    os.mkdir(actual_path)
    os.mkdir(picture_path)

    manager = Manager(actual_path)
    try:
        manager.start_recording()

        while True:
            time.sleep(1)
            manager.print_state()

    except KeyboardInterrupt:
        manager.stop_threads()
    finally:
        manager.convert_image()
        pass


if __name__ == "__main__":
    main()
