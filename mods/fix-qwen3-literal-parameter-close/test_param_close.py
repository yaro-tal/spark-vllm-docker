#!/usr/bin/env python3
"""Checks for mods/fix-qwen3-literal-parameter-close. Run inside the container,
after the mod has been applied:

    python3 test_param_close.py

Exits non-zero on the unpatched parser. Two properties per case:
  final      converter(raw, False) returns exactly the expected arguments.
  streaming  for every prefix of raw, each value from converter(prefix, True) is
             a prefix of that parameter's final value. The parser engine drops a
             call's arguments if a later snapshot does not extend an earlier one.
"""
import json
import sys

from vllm.parser import qwen3

conv = qwen3._qwen3_arg_converter
XML = '<root>\n  <parameter name="a">1</parameter>\n  <b>2</b>\n</root>'
CASES = [
    ("plain",
     "\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n",
     {"path": "/tmp/a.txt", "content": "hello"}),
    ("literal </parameter> mid-value",
     "\n<parameter=command>\necho '</parameter>' > x.xml && cat x.xml\n</parameter>\n"
     "<parameter=timeout>\n5\n</parameter>\n",
     {"command": "echo '</parameter>' > x.xml && cat x.xml", "timeout": "5"}),
    ("xml with <parameter ...> elements",
     f"\n<parameter=content>\n{XML}\n</parameter>\n", {"content": XML}),
    ("value ending in a literal </parameter>",
     "\n<parameter=content>\n<a>1</parameter>\n</parameter>\n", {"content": "<a>1</parameter>"}),
    ("missing final </parameter>",
     "\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=content>\nhello\n",
     {"path": "/tmp/a.txt", "content": "hello"}),
    ("indentation and blank line kept",
     "\n<parameter=new_string>\n    return x\n\n</parameter>\n", {"new_string": "    return x\n"}),
    ("code with < and >",
     "\n<parameter=code>\nif a < b and c > d: print('<ok>')\n</parameter>\n",
     {"code": "if a < b and c > d: print('<ok>')"}),
    ("no wrapping newlines",
     "<parameter=a>x</parameter><parameter=b>y</parameter>", {"a": "x", "b": "y"}),
    ("empty value", "\n<parameter=a>\n\n</parameter>\n", {"a": ""}),
]

failed = 0
for name, raw, want in CASES:
    got = json.loads(conv(raw, False))
    ok_final = got == want
    bad_at = None
    for n in range(len(raw) + 1):
        part = json.loads(conv(raw[:n], True))
        for k, v in part.items():
            if k not in want or not want[k].startswith(v):
                bad_at = (n, k, v)
                break
        if bad_at:
            break
    ok = ok_final and bad_at is None
    failed += not ok
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
    if not ok_final:
        print(f"       final: got {got!r}\n              want {want!r}")
    if bad_at:
        print(f"       streaming: at {bad_at[0]} chars, {bad_at[1]}={bad_at[2]!r} "
              f"is not a prefix of {want.get(bad_at[1])!r}")
sys.exit(1 if failed else 0)
