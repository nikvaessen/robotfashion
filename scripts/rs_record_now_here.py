#! /usr/bin/env python3

import os
import pathlib
import time

from queue import Queue

import lycon
import numpy as np

from rstools import (
    ImageProcessingThread,
    RecordingThread,
    generate_context,
    query_device_ids,
)


class Manager:
    def __init__(self, num_processing_threads=5):
        self.ctx = generate_context()
        time.sleep(1)

        self.device_ids = []
        self.processing_threads = []
        self.queue = Queue()
        self.recording_threads = []

        self.frame_delta = []

        self.num_processing_threads = num_processing_threads

    def start_recording(self):
        self.device_ids.clear()
        self.device_ids += query_device_ids(self.ctx)

        print("recognized {} devices: {}".format(len(self.device_ids), self.device_ids))

        if len(self.device_ids) == 0:
            exit(0)

        for _ in range(0, self.num_processing_threads):
            processing_thread = ImageProcessingThread(self.queue)
            processing_thread.start()

            self.processing_threads.append(processing_thread)

        for dev_id in self.device_ids:
            record_thread = RecordingThread(
                pathlib.Path(os.getcwd()), self.queue, self.ctx, dev_id
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

    @staticmethod
    def convert_image():
        for file in os.listdir(os.getcwd()):
            if "npz" in file:
                print(file)
                data = np.load(file)["data"]

                filename = file.split(".")[1] = ".png"
                lycon.save(filename, data)


def main():
    manager = Manager()

    try:
        manager.start_recording()

        while True:
            time.sleep(1)
            manager.print_state()

    except KeyboardInterrupt:
        manager.stop_threads()
    finally:
        manager.convert_image()


if __name__ == "__main__":
    main()
