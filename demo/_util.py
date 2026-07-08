"""Shared pretty-printing helpers so every service demo looks consistent
and is easy for a review panel to read on screen."""
import json, sys

G = "\033[92m"; C = "\033[96m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; X = "\033[0m"
USE_COLOR = sys.stdout.isatty()
def _c(s, col): return f"{col}{s}{X}" if USE_COLOR else s

def banner(title):
    line = "=" * 74
    print("\n" + _c(line, C))
    print(_c(f"  {title}", B))
    print(_c(line, C))

def step(msg):      print(_c(f"\n▶ {msg}", Y))
def show(label, v):
    if isinstance(v, (dict, list)):
        v = json.dumps(v, indent=2, default=str)
    print(f"   {_c(label, B)}: {v}")

def check(desc, cond):
    mark = _c("PASS", G) if cond else _c("FAIL", R)
    print(f"   [{mark}] {desc}")
    if not cond:
        raise AssertionError(desc)

def done(name):
    print(_c(f"\n✔ {name} — all checks passed\n", G))