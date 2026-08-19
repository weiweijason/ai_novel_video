"""Patch Wan to make flash_attn optional"""
import os

f = '/opt/venv/lib/python3.10/site-packages/wan/modules/attention.py'
if os.path.exists(f):
    with open(f, 'r') as fh:
        lines = fh.readlines()
    
    new_lines = []
    for line in lines:
        if line.strip() == 'import flash_attn':
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + 'try:\n')
            new_lines.append(indent + '    import flash_attn\n')
            new_lines.append(indent + 'except ImportError:\n')
            new_lines.append(indent + '    flash_attn = None\n')
        else:
            new_lines.append(line)
    
    with open(f, 'w') as fh:
        fh.writelines(new_lines)
    print("Patched attention.py to make flash_attn optional")
else:
    print("attention.py not found, skipping patch")
