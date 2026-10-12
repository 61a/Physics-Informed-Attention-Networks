"""Launch a command fully detached from the caller session (survives WorkBuddy
session end). Usage: python launch_detached.py <script.py> [args...]
Redirects stdout/stderr to <script>_detached.log.
"""
import os
import subprocess
import sys

PY = r"C:\ProgramData\anaconda3\envs\envly61\python.exe"
ROOT = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    print("usage: launch_detached.py <script.py> [args...]")
    sys.exit(1)

target = sys.argv[1]
args = sys.argv[2:]
script = os.path.join(ROOT, target)
logpath = os.path.join(ROOT, os.path.splitext(target)[0] + "_detached.log")
logf = open(logpath, "a", encoding="utf-8")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW

p = subprocess.Popen(
    [PY, "-u", script] + args,
    cwd=ROOT,
    stdout=logf, stderr=subprocess.STDOUT,
    creationflags=flags,
    close_fds=True,
)
print(f"Detached PID={p.pid} log={logpath}")
print(f"cmd: python -u {target} {' '.join(args)}")
