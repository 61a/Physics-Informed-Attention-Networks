"""Monitor all python processes every 3s; log existence to file.
Use to catch the exact moment a training process disappears (silent kill/crash).
Usage: monitor_all.py [--out monitor.log] [--duration 900]"""
import ctypes
import ctypes.wintypes as wt
import time
import sys
import os
import argparse
import subprocess


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.POINTER(ctypes.c_ulong)), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', ctypes.c_long), ('dwFlags', wt.DWORD),
                ('szExeFile', ctypes.c_wchar * 260)]


def python_pids():
    TH32CS_SNAPPROCESS = 2
    snap = ctypes.windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == -1:
        return []
    pe = PROCESSENTRY32W()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    ok = ctypes.windll.kernel32.Process32FirstW(snap, ctypes.byref(pe))
    out = []
    while ok:
        if 'python' in pe.szExeFile.lower():
            out.append(pe.th32ProcessID)
        ok = ctypes.windll.kernel32.Process32NextW(snap, ctypes.byref(pe))
    ctypes.windll.kernel32.CloseHandle(snap)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="monitor.log")
    ap.add_argument("--duration", type=int, default=1200)
    args = ap.parse_args()

    log = open(args.out, "a", encoding="utf-8")
    t_end = time.time() + args.duration
    seen = set()
    gpu_proc = None
    try:
        gpu_proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    while time.time() < t_end:
        pids = python_pids()
        ts = time.strftime("%H:%M:%S")
        gpu = "?"
        if gpu_proc is not None:
            try:
                out = gpu_proc.communicate(timeout=2)[0].decode().strip()
                gpu = out.replace("\n", " ")
            except Exception:
                pass
            gpu_proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        line = f"{ts} python_procs={pids} gpu={gpu}"
        if set(pids) != seen:
            line += "  <-- CHANGED"
        log.write(line + "\n")
        log.flush()
        seen = set(pids)
        time.sleep(4)
    log.close()
    print("monitor done", flush=True)


if __name__ == "__main__":
    main()
