import pyrealsense2 as rs

import time


def generate_context():
    return rs.context()


def query_device_ids(ctx: rs.context):
    while True:
        time.sleep(1)

        print(
            [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
        )
