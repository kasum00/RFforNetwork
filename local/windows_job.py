"""Windows Job Object: supervisor handle lifetime owns the entire worker tree."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class WindowsJob:
    """No non-Windows fallback. Assignment failure blocks all training.

    Uses CPython Popen's retained Windows process handle, not a racy fresh PID
    lookup. A child must wait for job_assignment.json before authorization/fit.
    """
    def __init__(self, *, kernel=None):
        self.kernel, self.handle = kernel, None

    def __enter__(self):
        if os.name != "nt":
            raise PermissionError("Windows Job Object requires Windows")
        if self.kernel is None:
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            k = self.kernel
            k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k.CreateJobObjectW.restype = wintypes.HANDLE
            k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k.SetInformationJobObject.restype = wintypes.BOOL
            k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k.AssignProcessToJobObject.restype = wintypes.BOOL
            k.CloseHandle.argtypes = [wintypes.HANDLE]
            k.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("CreateJobObjectW failed; training blocked")
        limits = EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError("SetInformationJobObject failed; training blocked")
        return self

    def assign(self, process):
        handle = getattr(process, "_handle", None)
        if handle is None or not self.kernel.AssignProcessToJobObject(self.handle, int(handle)):
            raise OSError("Windows Job assignment failed; no unprotected worker is permitted")

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            if not self.kernel.CloseHandle(handle):
                raise OSError("CloseHandle(Job) failed; manual process review required")

    def __exit__(self, *exc):
        self.close()
