#!/usr/bin/env python3
"""LIBERO program backend restricted to a frozen bundle: run_program accepts only the bundle's program texts.

The bundle is the directory named by REA_PROGRAM_BUNDLE (tools/build_program_bundle.py writes it).
"""
import importlib.util
import ctypes
import json
import os
from pathlib import Path
import sys

source = Path(os.environ.get('REA_ASPIRE_BACKEND', str(Path(__file__).with_name('aspire_backend.py'))))
spec = importlib.util.spec_from_file_location('rea_aspire_backend_base', source)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def configure_egl_cuda_lane():
    """Render on the single CUDA lane despite a different EGL device order."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit():
        raise ValueError('LIBERO rendering requires one numeric CUDA_VISIBLE_DEVICES lane')

    # This robosuite version checks that the two environment indices match at
    # import time, although CUDA and EGL enumerate this host in different orders.
    # Import using the CUDA lane, then override only its display factory below.
    previous_egl_device = os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    try:
        from robosuite.renderers.context import egl_context
    finally:
        if previous_egl_device is not None:
            os.environ['MUJOCO_EGL_DEVICE_ID'] = previous_egl_device

    egl = ctypes.CDLL('libEGL.so.1')
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query_cuda_device = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t),
    )(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
    egl_index = None
    for index, device in enumerate(egl_context.EGL.eglQueryDevicesEXT()):
        cuda_device = ctypes.c_ssize_t()
        # EGL_CUDA_DEVICE_NV returns the ordinal within CUDA_VISIBLE_DEVICES.
        if query_cuda_device(device, 0x323A, ctypes.byref(cuda_device)) and cuda_device.value == 0:
            egl_index = index
            break
    if egl_index is None:
        raise RuntimeError(f'No EGL device matches CUDA_VISIBLE_DEVICES={visible}')

    original_factory = egl_context.create_initialized_egl_device_display

    def initialize_cuda_lane_display(device_id=0):
        previous = os.environ.get('MUJOCO_EGL_DEVICE_ID')
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(egl_index)
        try:
            return original_factory(device_id=device_id)
        finally:
            if previous is None:
                os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
            else:
                os.environ['MUJOCO_EGL_DEVICE_ID'] = previous

    egl_context.create_initialized_egl_device_display = initialize_cuda_lane_display
    print(f'EGL renderer: CUDA_VISIBLE_DEVICES={visible} -> EGL device {egl_index}', file=sys.stderr, flush=True)
    return {'physicalGpu': int(visible), 'cudaVisibleDevices': visible, 'eglDeviceId': egl_index}


class ProgramLiberoBackend(base.AspireBackend):
    def __init__(self, args, **kwargs):
        root = Path(os.environ['REA_PROGRAM_BUNDLE'])
        bundle = json.loads((root / 'bundle.json').read_text())
        self.frozen_programs = {
            (task['suite'], task['task']): [(root / item['path']).read_text() for item in task['candidates'].values()]
            for task in bundle['tasks']
        }
        super().__init__(args, **kwargs)
        self.gpu_assignment = None
        if kwargs.get('env_factory') is None:
            self.gpu_assignment = configure_egl_cuda_lane()

    def reset(self, request):
        reply = super().reset(request)
        reply['gpuAssignment'] = self.gpu_assignment
        return reply

    def _check_program_pin(self, _allowed, code, _unused):
        candidates = self.frozen_programs[(self.task['suite'], self.task['name'])]
        if code not in candidates:
            raise ValueError('Program differs from the frozen candidates for this task')

    def _run_reply(self, *args, **kwargs):
        reply = super()._run_reply(*args, **kwargs)
        fields = (
            'which', 'steps_before', 'steps_after', 'steps', 'wall_s', 'exception', 'exception_class',
            'truncated', 'truncated_reason', 'horizon', 'stdout_tail', 'stderr_tail', 'wall_timeout_s',
            'wall_used_s', 'wall_budget_s', 'wall_left_s', 'infra_probe',
        )
        return {key: reply[key] for key in fields}


if __name__ == '__main__':
    base.AspireBackend = ProgramLiberoBackend
    base.main()
