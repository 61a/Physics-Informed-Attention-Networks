"""List processes matching keywords (Huorong, wsctrl, Hips, defender etc)."""
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
kw = ['huorong', 'hips', 'wsctrl', 'usysdiag', 'defender', 'msmpeng', 'security',
      'asw', 'avp', '360', 'qqpctray', 'kxescore', 'kwsprotect']
while ok:
    exe = pe.szExeFile
    low = exe.lower()
    for k in kw:
        if k in low:
            print(f'PID={pe.th32ProcessID} PPID={pe.th32ParentProcessID} {exe}')
            break
    ok = ctypes.windll.kernel32.Process32NextW(snap, ctypes.byref(pe))
ctypes.windll.kernel32.CloseHandle(snap)
