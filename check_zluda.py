import os
import sys
import struct
import ctypes
import subprocess
import platform
import traceback

def pe_machine(path):
    with open(path,'rb') as f:
        f.seek(0x3c)
        e_lfanew = struct.unpack('<I', f.read(4))[0]
        f.seek(e_lfanew + 4)
        return struct.unpack('<H', f.read(2))[0]
def machine_name(m):
    return {0x8664:'x64',0x014c:'x86',0xAA64:'arm64'}.get(m,str(m))
print("python exe", sys.executable)
print("python arch", platform.architecture()[0], "pointer bits", struct.calcsize('P')*8)
print("exec_prefix", sys.exec_prefix)
candidates = []
candidates.append(os.path.join(sys.exec_prefix,'DLLs','nvcuda.dll'))
candidates.append(os.path.join(os.path.dirname(sys.executable),'nvcuda.dll'))
for p in os.environ.get('PATH','').split(os.pathsep):
    p = p.strip()
    if not p:
        continue
    f = os.path.join(p,'nvcuda.dll')
    if os.path.exists(f):
        candidates.append(f)
seen = []
for p in candidates:
    if p and os.path.exists(p) and p not in seen:
        seen.append(p)
print("nvcuda.dll found", seen or 'NONE')
for p in seen:
    try:
        m = pe_machine(p)
    except Exception as e:
        m = None
    print("path", p, "size", os.path.getsize(p), "machine", machine_name(m))
    try:
        lib = ctypes.WinDLL(p)
        print("ctypes.WinDLL load OK", p)
        del lib
    except OSError as e:
        print("ctypes.WinDLL failed", repr(e))
        code = getattr(e,'winerror', None) or ctypes.get_last_error()
        try:
            msg = ctypes.FormatError(code) if code else None
        except Exception:
            msg = None
        print("WinError", code, "message", msg)
print("checking amdhip64.dll in PATH and System32")
hip = []
for p in os.environ.get('PATH','').split(os.pathsep):
    f = os.path.join(p,'amdhip64.dll')
    if os.path.exists(f):
        hip.append(f)
sys32 = os.path.join(os.environ.get('SYSTEMROOT','C:\\Windows'),'System32','amdhip64.dll')
if os.path.exists(sys32):
    hip.append(sys32)
print("amdhip64.dll found", hip or 'NONE')
print("checking common MSVC runtimes in PATH")
rts = ['vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll']
found = []
for r in rts:
    for p in os.environ.get('PATH','').split(os.pathsep):
        f = os.path.join(p, r)
        if os.path.exists(f):
            found.append(f)
print("msvc runtimes found", found or 'NONE')
print("attempting to import torch within this process")
try:
    import torch
    print("torch.version.cuda", getattr(torch.version,'cuda', None))
    print("torch.cuda.is_available()", torch.cuda.is_available())
except Exception as e:
    print("import torch raised", repr(e))
    try:
        proc = subprocess.run([sys.executable, '-c', 'import torch; print(getattr(torch.version,\"cuda\",None)); print(torch.cuda.is_available())'], capture_output=True, text=True)
        print("subprocess stdout")
        print(proc.stdout)
        print("subprocess stderr")
        print(proc.stderr)
    except Exception as e2:
        print("subprocess import attempt failed", repr(e2))
print("END OF CHECK")
