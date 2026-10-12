"""List python-related processes using ctypes (no psutil needed)."""
import ctypes
import ctypes.wintypes as wt


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.POINTER(ctypes.c_ulong)), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', ctypes.c_long), ('dwFlags', wt.DWORD),
                ('szExeFile', ctypes.c_wchar * 260)]


TH32CS_SNAPPROCESS = 2
snap = ctypes.windll.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
if snap == -1:
    print('snapshot failed')
    raise SystemExit(1)

pe = PROCESSENTRY32W()
pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
ok = ctypes.windll.kernel32.Process32FirstW(snap, ctypes.byref(pe))
procs = []
while ok:
    exe = pe.szExeFile
    if 'python' in exe.lower():
        procs.append((pe.th32ProcessID, pe.th32ParentProcessID, exe))
    ok = ctypes.windll.kernel32.Process32NextW(snap, ctypes.byref(pe))
ctypes.windll.kernel32.CloseHandle(snap)

print(f'python procs: {len(procs)}')
for pid, ppid, name in procs:
    print(f'  PID={pid} PPID={ppid} {name}')
