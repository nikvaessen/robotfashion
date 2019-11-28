import pyrealsense2 as rs

import time


def query_device_ids(ctx: rs.context):
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
