#!/usr/bin/env python3
"""Reads a frame trace (SOTC_FRAME_TRACE, src/diag/frame_trace.cpp) and shows what the threads
waited for, frame by frame.

    tools/frame_trace.py [frame_trace.bin] [--frames N] [--lanes K] [--bucket MS]

Prints:
  1. per thread: time blocked per second, by wait kind and game caller (the largest ones);
  2. per frame (time between two presents): the frame time and, for the K frames in the
     middle of the window, a lane per thread (one character per bucket of time):
        '.'  blocked in a wait of the game (event flag, queue, cond, mutex, sleep)
        'R'  reading GPU data back (a fault waiting for the GPU layer)
        'w'  stage A waiting for stage B
        ' '  stage B with no packets
        '#'  running, or nothing known
     with the timeline marks above the lanes (S submit, F flip, P present, G GPU busy);
  3. per frame, the critical path backwards from the flip submission: the waits that held the
     submitting thread back, and which thread released each.
"""

import argparse
import collections
import os
import struct
import sys

WAIT_NAMES = {1: "equeue", 2: "eventflag", 3: "cond", 4: "condtimed", 5: "mutex", 6: "usleep",
              7: "nanosleep"}
EVENT = struct.Struct("<QQQIHcB")


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"SOTCTRC1":
        sys.exit(f"{path}: not a frame trace")
    count, = struct.unpack_from("<I", data, 8)
    offset = 16
    threads = []
    for _ in range(count):
        tid, _ = struct.unpack_from("<II", data, offset)
        name = data[offset + 8:offset + 24].split(b"\0")[0].decode(errors="replace")
        threads.append((tid, name))
        offset += 24
    events_count, = struct.unpack_from("<Q", data, offset)
    offset += 8
    events = []
    for i in range(events_count):
        ns, obj, caller, caller2, thread, kind, sub = EVENT.unpack_from(data, offset + i * 32)
        if ns:
            events.append((ns, kind.decode(), sub, obj, caller, caller2, thread))
    events.sort(key=lambda e: e[0])
    return threads, events


def thread_label(threads, index):
    tid, name = threads[index]
    return f"{name}/{tid}"


def main():
    parser = argparse.ArgumentParser()
    default = os.path.expanduser("~/.local/share/sotcport/frame_trace.bin")
    parser.add_argument("path", nargs="?", default=default)
    parser.add_argument("--lanes", type=int, default=3, help="frames drawn as lanes")
    parser.add_argument("--bucket", type=float, default=0.1, help="ms per character")
    parser.add_argument("--paths", type=int, default=6, help="frames with a critical path")
    args = parser.parse_args()

    threads, events = load(args.path)
    t0 = events[0][0]
    span = (events[-1][0] - t0) / 1e9
    print(f"{len(events)} events, {len(threads)} threads, {span:.2f} s")

    # Wait intervals per thread: 'B' and the next 'E' of the same kind and object.
    open_waits = {}
    waits = collections.defaultdict(list)  # thread -> [(begin, end, kind, obj, caller, caller2)]
    wakes = collections.defaultdict(list)  # (kind, obj) -> [(ns, thread, caller)]
    readbacks = collections.defaultdict(list)  # thread -> [(begin, end)]
    open_readbacks = {}
    marks = collections.defaultdict(list)  # letter -> [(ns, thread, value)]
    for ns, kind, sub, obj, caller, caller2, thread in events:
        if kind == "B":
            open_waits[(thread, sub, obj)] = (ns, caller, caller2)
        elif kind == "E":
            begin = open_waits.pop((thread, sub, obj), None)
            if begin:
                waits[thread].append((begin[0], ns, sub, obj, begin[1], begin[2]))
        elif kind == "K":
            wakes[(sub, obj)].append((ns, thread, caller))
        elif kind == "M":
            letter = chr(sub)
            marks[letter].append((ns, thread, obj))
            if letter == "R":
                open_readbacks[thread] = ns
            elif letter == "r" and thread in open_readbacks:
                readbacks[thread].append((open_readbacks.pop(thread), ns))

    # 1. Blocked time per thread.
    print("\nBlocked time per thread (ms/s), largest callers")
    rows = []
    for thread, intervals in waits.items():
        total = sum(e - b for b, e, *_ in intervals) / 1e6 / span
        by_caller = collections.Counter()
        for b, e, sub, obj, caller, caller2 in intervals:
            by_caller[(WAIT_NAMES.get(sub, sub), caller, caller2)] += (e - b) / 1e6 / span
        rows.append((total, thread, by_caller))
    for thread, intervals in readbacks.items():
        total = sum(e - b for b, e in intervals) / 1e6 / span
        rows.append((total, thread, collections.Counter({("readback", 0, 0): total})))
    for total, thread, by_caller in sorted(rows, key=lambda r: -r[0]):
        if total < 5:
            continue
        top = ", ".join(f"{k} {c:#x}<{c2:#x} {v:.0f}" for (k, c, c2), v in by_caller.most_common(4))
        print(f"  {thread_label(threads, thread):28s} {total:6.0f}  {top}")

    # 2. Frames.
    presents = [ns for ns, _, _ in marks["P"]]
    frames = list(zip(presents, presents[1:]))
    if not frames:
        print("no presents in the window")
        return
    times = sorted((e - b) / 1e6 for b, e in frames)
    print(f"\n{len(frames)} frames: median {times[len(times) // 2]:.2f} ms, "
          f"p90 {times[int(len(times) * 0.9)]:.2f} ms, max {times[-1]:.2f} ms")

    def first_after(letter, t, thread=None):
        for ns, th, value in marks[letter]:
            if ns >= t and (thread is None or th == thread):
                return ns, th, value
        return None

    names = {i: n for i, (_, n) in enumerate(threads)}
    pipeline = {i for i, n in names.items() if n in ("shadPS4:GpuComm", "sotc:DrawRec")}
    stage_a = next((i for i, n in names.items() if n == "shadPS4:GpuComm"), None)
    stage_b = next((i for i, n in names.items() if n == "sotc:DrawRec"), None)

    def lanes(begin, end):
        width = int((end - begin) / 1e6 / args.bucket) + 1
        bucket = args.bucket * 1e6

        def col(t):
            return min(width - 1, max(0, int((t - begin) / bucket)))

        lines = []
        # Marks row: S, F, P; GPU row from 'g'/'G' spans.
        row = [" "] * width
        for letter in "SFP":
            for ns, _, _ in marks[letter]:
                if begin <= ns < end:
                    row[col(ns)] = letter
        lines.append(("marks", "".join(row)))
        row = [" "] * width
        starts = [ns for ns, _, _ in marks["g"]]
        ends = [ns for ns, _, _ in marks["G"]]
        for s in starts:
            e = next((x for x in ends if x >= s), None)
            if e is None or e < begin or s > end:
                continue
            for c in range(col(s), col(e) + 1):
                row[c] = "G"
        lines.append(("GPU", "".join(row)))
        active = set()
        for th in waits:
            if any(b < end and e > begin for b, e, *_ in waits[th]):
                active.add(th)
        for th in readbacks:
            if any(b < end and e > begin for b, e in readbacks[th]):
                active.add(th)
        active |= pipeline
        for th in sorted(active, key=lambda t: names[t]):
            row = ["#"] * width
            for b, e, *_ in waits.get(th, []):
                if b < end and e > begin:
                    for c in range(col(b), col(e) + 1):
                        row[c] = "."
            for b, e in readbacks.get(th, []):
                if b < end and e > begin:
                    for c in range(col(b), col(e) + 1):
                        row[c] = "R"
            if th == stage_a:
                pending = None
                ws = sorted([(ns, "w") for ns, t2, _ in marks["w"] if t2 == th] +
                            [(ns, "W") for ns, t2, _ in marks["W"] if t2 == th])
                for ns, letter in ws:
                    if letter == "w":
                        pending = ns
                    elif pending is not None:
                        if pending < end and ns > begin:
                            for c in range(col(pending), col(ns) + 1):
                                row[c] = "w"
                        pending = None
            if th == stage_b:
                es = sorted([(ns, "e") for ns, t2, _ in marks["e"] if t2 == th] +
                            [(ns, "b") for ns, t2, _ in marks["b"] if t2 == th])
                pending = None
                for ns, letter in es:
                    if letter == "e":
                        pending = ns
                    elif pending is not None:
                        if pending < end and ns > begin:
                            for c in range(col(pending), col(ns) + 1):
                                row[c] = " "
                        pending = None
            lines.append((thread_label(threads, th), "".join(row)))
        return lines

    middle = len(frames) // 2
    for b, e in frames[middle:middle + args.lanes]:
        print(f"\nframe {(e - b) / 1e6:.2f} ms ({args.bucket} ms per character)")
        for label, row in lanes(b, e):
            print(f"  {label:28s} |{row}|")

    # 3. Critical path backwards from each flip submission ('F').
    def wait_before(thread, t):
        best = None
        for w in waits.get(thread, []):
            if w[1] <= t + 1000 and (best is None or w[1] > best[1]):
                best = w
        rb = None
        for r in readbacks.get(thread, []):
            if r[1] <= t + 1000 and (rb is None or r[1] > rb[1]):
                rb = r
        if rb and (best is None or rb[1] > best[1]):
            return ("readback", rb[0], rb[1], None)
        if best:
            return ("wait", best[0], best[1], best)
        return None

    def waker(w):
        b, e, sub, obj, caller, caller2 = w
        candidates = [k for k in wakes.get((sub, obj), []) if b - 1000 <= k[0] <= e + 20000]
        return candidates[-1] if candidates else None

    print("\nCritical path back from the flip submission (newest first)")
    for b, e in frames[middle:middle + args.paths]:
        flip = first_after("F", b)
        if not flip or flip[0] > e:
            continue
        t, thread = flip[0], flip[1]
        steps = [f"F at {(t - b) / 1e6:.2f} on {names[thread]}"]
        for _ in range(8):
            found = wait_before(thread, t)
            if not found or found[2] < b - 5e6:
                break
            kind, wb, we, w = found
            run = (t - we) / 1e6
            if kind == "readback":
                steps.append(f"ran {run:.2f}, readback {(we - wb) / 1e6:.2f}")
                t = wb
                continue
            name = WAIT_NAMES.get(w[2], w[2])
            k = waker(w)
            if k and k[1] != thread:
                steps.append(f"ran {run:.2f}, {name}@{w[4]:#x}<{w[5]:#x} {(we - wb) / 1e6:.2f} "
                             f"(woken by {names[k[1]]} @{k[2]:#x})")
                t, thread = k[0], k[1]
            else:
                steps.append(f"ran {run:.2f}, {name}@{w[4]:#x}<{w[5]:#x} {(we - wb) / 1e6:.2f}")
                t = wb
        print(f"  frame {(e - b) / 1e6:.2f} ms: " + " <- ".join(steps))


if __name__ == "__main__":
    main()
