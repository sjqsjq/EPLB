from typing import Optional

_global_controller = None

def get_pb_oeplb_controller():
    return _global_controller

def set_pb_oeplb_controller(ctrl):
    global _global_controller
    _global_controller = ctrl
