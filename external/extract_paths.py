"""Extract vector paths from a matplotlib-generated PDF content stream."""
import re, zlib, math

def content_stream(path):
    raw = open(path, "rb").read()
    for m in re.finditer(rb"stream\r?\n", raw):
        s = m.end(); e = raw.find(b"endstream", s)
        try: d = zlib.decompress(raw[s:e])
        except Exception: continue
        if b" m\n" in d[:2000] or b"re W n" in d[:4000]:
            return d.decode("latin-1")
    raise RuntimeError("no content stream")

def mat_mul(a, b):
    return (a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
            a[4]*b[0]+a[5]*b[2]+b[4], a[4]*b[1]+a[5]*b[3]+b[5])

def apply(m, x, y):
    return (m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5])

def bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(1, n+1):
        t = i/n; u = 1-t
        out.append((u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                    u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]))
    return out

def parse(txt):
    toks = re.findall(r"\[[^\]]*\]|[^\s\[\]]+", txt)
    ctm = (1,0,0,1,0,0); stack = []
    stroke = (0,0,0); fill = (0,0,0); dash = "solid"; lw = 1.0
    cur = []; subpaths = []; start = None; pos = None
    out = []
    args = []
    def flush(kind):
        nonlocal cur, subpaths
        if cur: subpaths.append(cur)
        if subpaths:
            out.append(dict(kind=kind, color=stroke if kind=="S" else fill,
                            dash=dash, lw=lw, subpaths=subpaths))
        cur = []; subpaths = []
    for t in toks:
        try:
            args.append(float(t)); continue
        except ValueError:
            pass
        if t.startswith("["):
            args.append(t); continue
        op = t
        if op == "q": stack.append((ctm, stroke, fill, dash, lw))
        elif op == "Q":
            if stack: ctm, stroke, fill, dash, lw = stack.pop()
        elif op == "cm" and len(args) >= 6:
            ctm = mat_mul(tuple(args[-6:]), ctm)
        elif op == "m" and len(args) >= 2:
            if cur: subpaths.append(cur)
            pos = apply(ctm, args[-2], args[-1]); start = pos; cur = [pos]
        elif op == "l" and len(args) >= 2:
            pos = apply(ctm, args[-2], args[-1]); cur.append(pos)
        elif op == "c" and len(args) >= 6:
            p1 = apply(ctm, args[-6], args[-5]); p2 = apply(ctm, args[-4], args[-3])
            p3 = apply(ctm, args[-2], args[-1])
            if pos: cur.extend(bezier(pos, p1, p2, p3))
            pos = p3
        elif op == "v" and len(args) >= 4:
            p2 = apply(ctm, args[-4], args[-3]); p3 = apply(ctm, args[-2], args[-1])
            if pos: cur.extend(bezier(pos, pos, p2, p3))
            pos = p3
        elif op == "y" and len(args) >= 4:
            p1 = apply(ctm, args[-4], args[-3]); p3 = apply(ctm, args[-2], args[-1])
            if pos: cur.extend(bezier(pos, p1, p3, p3))
            pos = p3
        elif op == "re" and len(args) >= 4:
            x, y, w, h = args[-4:]
            pts = [apply(ctm, x, y), apply(ctm, x+w, y),
                   apply(ctm, x+w, y+h), apply(ctm, x, y+h)]
            if cur: subpaths.append(cur)
            subpaths.append(pts + [pts[0]]); cur = []
        elif op == "h":
            if cur and start: cur.append(start)
        elif op in ("S", "s"): flush("S")
        elif op in ("f", "F", "f*"): flush("f")
        elif op in ("B", "B*", "b", "b*"): flush("S")
        elif op == "n": cur = []; subpaths = []
        elif op == "RG" and len(args) >= 3: stroke = tuple(args[-3:])
        elif op == "rg" and len(args) >= 3: fill = tuple(args[-3:])
        elif op in ("G",) and len(args) >= 1: stroke = (args[-1],)*3
        elif op in ("g",) and len(args) >= 1: fill = (args[-1],)*3
        elif op == "w" and len(args) >= 1: lw = args[-1]
        elif op == "d" and len(args) >= 2:
            arr = args[-2]
            dash = "solid" if (isinstance(arr,str) and arr.strip("[] ") == "") else "dashed"
        args = []
    return out

if __name__ == "__main__":
    import sys
    paths = parse(content_stream(sys.argv[1]))
    print(f"{len(paths)} painted paths")
    from collections import Counter
    c = Counter((p["kind"], tuple(round(v,3) for v in p["color"]), p["dash"]) for p in paths)
    for k, n in c.most_common(12):
        print("  ", k, n)
