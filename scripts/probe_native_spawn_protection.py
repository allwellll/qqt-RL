#!/usr/bin/env python3
"""Trace v1.3.2's obfuscated relive state entry (requires pefile and unicorn).

Usage: python probe_native_spawn_protection.py <Client.exe>
Only executes a local PE image; no Windows process or client network is started.
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import pefile
import unicorn
from unicorn.x86_const import UC_X86_REG_EAX, UC_X86_REG_ECX, UC_X86_REG_EIP, UC_X86_REG_ESP

client = Path(sys.argv[1])
data = client.read_bytes()
expected = '670af4f24b3db6bac030a61d628cedc8e4522468f4ef4cb88a2239ad6b946f74'
assert hashlib.sha256(data).hexdigest() == expected, 'probe addresses require the official v1.3.2 client'
pe = pefile.PE(data=data)
image = pe.get_memory_mapped_image()
result = {'client_sha256': expected, 'birth_call': '0x5aff67 push 0xbb8; 0x5aff72 call 0x5b5202',
          'relive_handler': '0x603770 schema 0x0fba -> 0x5c0b08 -> vtable+0x28 -> 0x5c0edd'}
assert image[0x5aff67 - 0x400000:0x5aff6c - 0x400000] == bytes.fromhex('68b80b0000')
for prior_state in (0, 9):
    cpu = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_32)
    cpu.mem_map(0x400000, (len(image) + 4095) // 4096 * 4096)
    cpu.mem_write(0x400000, image)
    for start in (0, 0x2000000, 0x3000000):
        cpu.mem_map(start, 0x10000)
    cpu.mem_write(0x18, struct.pack('<I', 0x1000))
    cpu.mem_write(0x1008, struct.pack('<I', 0x2003000))
    cpu.reg_write(UC_X86_REG_ESP, 0x2008000)
    cpu.mem_write(0x2008000, struct.pack('<III', 0x3001000, 0x3002000, 0x3003000))
    cpu.reg_write(UC_X86_REG_ECX, 0x3002000)
    trace = []
    found = []

    def on_instruction(uc, address, size, _):
        trace.append(address)
        if address == 0x5c0bda:
            # The state manager getter would dereference its Windows-owned vtable.
            # Stub only its previous-state ID; execute timer setup unchanged.
            esp = uc.reg_read(UC_X86_REG_ESP)
            ret = struct.unpack('<I', uc.mem_read(esp, 4))[0]
            uc.reg_write(UC_X86_REG_EAX, prior_state)
            uc.reg_write(UC_X86_REG_ESP, esp + 4)
            uc.reg_write(UC_X86_REG_EIP, ret)
        if address == 0x5b5202:
            duration = struct.unpack('<I', uc.mem_read(uc.reg_read(UC_X86_REG_ESP) + 4, 4))[0]
            found.append({'duration_ms': duration, 'actor_offset': hex(uc.reg_read(UC_X86_REG_ECX) - 0x3003000),
                          'instructions': len(trace)})
            uc.emu_stop()

    cpu.hook_add(unicorn.UC_HOOK_CODE, on_instruction)
    cpu.emu_start(0x5c0edd, 0x3001000, count=200000)
    assert found and found[0]['duration_ms'] == 3000 and found[0]['actor_offset'] == '0x318', (prior_state, found)
    result[f'relive_prior_state_{prior_state}'] = found[0]
print(json.dumps(result, indent=2))
