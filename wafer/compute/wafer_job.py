#!/usr/bin/env python3
"""wafer-job: the wafer launch/supervision adapter on compute-jobs (task 940).

Four layers stay separate:
  samcloud registry   - the only physical admission/lease authority (resource wafer-services/gpu-metal)
  compute-jobs        - durable intent, exact claim, wait/reconnect/renew/return (pinned 9d56934, Jobs API)
  THIS adapter        - starts and observes the exact owned execution, its descendants and artifacts
  the workload        - an ordinary command (compress.py, a sequence builder, ...)

Lifecycle of `wafer-job run KEY --purpose P --exclusive -- CMD...`:
  1. jobs.acquire (durable intent before POST) and wait for the exact lease to be active.
  2. Headroom gate (deployment policy, not physical capacity): wait until macOS reports at least
     --min-free-pct system-wide free memory, re-reading after admission. Never kills anything.
  3. jobs.begin(KEY, execution=<job dir>) BEFORE spawning: a second begin refuses, so nothing relaunches.
  4. A detached supervisor (own session) starts CMD in its own session with WAFER_JOB=KEY in its
     environment. Every descendant inherits the marker, which is how ownership is observed.
  5. The supervisor renews every ttl/3, samples memory pressure and the owned processes' RSS, records
     the exit, then asks the observer for fresh absence and releases the exact claim with that evidence.

Observation (`wafer-job observe KEY`): scans `ps eww` for processes whose environment carries WAFER_JOB=KEY
(and records pid+start time). Answers gone only when ps succeeds and none are left; alive when any
are; unknown when ps fails or the record is incomplete. alive/unknown both retain the claim.

Recovery (`wafer-job attach KEY`): if the recorded supervisor is alive, report. Otherwise start a new
supervisor in adopt mode: it NEVER relaunches the command, it renews, waits for the owned processes to be
gone, then releases. If the claim expired meanwhile, the pool's expiry_policy=hold keeps it charged
(no expiry-as-absence) until this settles it.

Records: ~/var/wafer-jobs/KEY/ (0700): intent.json, exec.json, samples.jsonl, exit.json, release.json,
supervisor.log, out.log. No tokens. The compute-jobs record lives in its own store (COMPUTE_JOBS_HOME).
"""
import argparse, datetime, json, os, re, signal, subprocess, sys, time
from pathlib import Path

from compute_jobs.jobs import Jobs
from compute_jobs.registry import Registry
from compute_jobs.state import Store, Conflict

VERSION = "0.1.0"
RESOURCE = "wafer-services/gpu-metal"
JOBS_ROOT = Path(os.environ.get("WAFER_JOBS_ROOT", Path.home() / "var/wafer-jobs"))
CJ_HOME = os.environ.get("COMPUTE_JOBS_HOME", str(Path.home() / ".local/state/samcloud/compute-jobs"))
MARK = "WAFER_JOB"
SUP_MARK = "WAFER_JOB_SUPERVISOR"


def now():
    return time.time()


def iso(t=None):
    return datetime.datetime.fromtimestamp(t or now(), datetime.timezone.utc).isoformat(timespec="seconds")


API = os.environ.get("SAMCLOUD_API", "https://cloud.samtg.xyz").rstrip("/")
API = API if API.endswith("/api/v1") else API + "/api/v1"     # seats export the bare origin


def jobs():
    return Jobs(Store(CJ_HOME), Registry(API))


def jdir(key):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        raise ValueError("key must be [A-Za-z0-9_.-]+")
    return JOBS_ROOT / key


def wjson(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o600); os.replace(tmp, path)


def rjson(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def log(key, msg):
    with open(jdir(key) / "supervisor.log", "a") as f:
        f.write(f"{iso()} {msg}\n")


# ---------- measurement (observational only) ------------------------------------------------------
def free_pct():
    """macOS system-wide free memory percentage (memory_pressure); None if unreadable."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def swap_used_mb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def owned(key, marker=MARK):
    """Processes whose environment carries marker=KEY. Returns (list|None); None means observation failed."""
    try:
        r = subprocess.run(["ps", "eww", "-A", "-o", "pid=,rss=,lstart=,command="], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    me = os.getpid(); pat = re.compile(r"(^|\s)%s=%s(\s|$)" % (marker, re.escape(key)))
    rows = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        pid, rss = int(parts[0]), int(parts[1]); start = " ".join(parts[2:7]); cmd = parts[7]
        if pid == me or not pat.search(cmd):
            continue
        rows.append({"pid": pid, "rss_kb": rss, "start": start, "cmd": cmd.split(" %s=" % marker)[0].split(" PATH=")[0][:200]})
    return rows


def observe(key):
    d = jdir(key); ex = rjson(d / "exec.json")
    ev = {"execution": str(d), "observed_at": now(), "observer": f"wafer-job {VERSION} ps-env-marker"}
    if ex is None:
        # begin was fenced but the supervisor never recorded a spawn: still unknown, not gone.
        ev["state"] = "unknown"; ev["why"] = "no exec record"
        return ev
    procs = owned(key)
    if procs is None:
        ev["state"] = "unknown"; ev["why"] = "ps failed"
    else:
        ev["state"] = "alive" if procs else "gone"; ev["pids"] = [p["pid"] for p in procs]
    return ev


# ---------- lifecycle --------------------------------------------------------------------------------
def cmd_run(a):
    d = jdir(a.key); d.mkdir(parents=True, exist_ok=True); os.chmod(d, 0o700)
    intent = {"key": a.key, "resource": RESOURCE, "purpose": a.purpose, "argv": a.argv, "cwd": os.path.abspath(a.cwd),
              "exclusive": a.exclusive, "memory_mb": a.memory, "min_free_pct": a.min_free_pct, "adapter": VERSION,
              "created_at": iso()}
    old = rjson(d / "intent.json")
    if old and (old["argv"] != intent["argv"] or old["cwd"] != intent["cwd"]):
        raise Conflict("this key already names a different command; use a new key for new work")
    if not old:
        wjson(d / "intent.json", intent)
    J = jobs()
    row = J.acquire(a.key, RESOURCE, a.purpose, memory=a.memory, exclusive=a.exclusive, service=a.service,
                    ttl=a.ttl, max_total=a.max_total)
    print(json.dumps({"admission": row["state"], "lease": row["lease_id"]}), flush=True)
    while row["state"] in ("queued", "admitting"):
        row = J.refresh(a.key, wait=True)
    if row["state"] != "active":
        raise Conflict(f"not admitted: {row['state']}")
    if row["workload"] != "not_started":
        print(json.dumps({"note": "already begun: reconnecting, never relaunching"}), flush=True)
        return cmd_attach(a)
    # Headroom gate: deployment policy over a fresh physical observation. Wait, never kill.
    while True:
        f = free_pct()
        if f is not None and f >= a.min_free_pct:
            break
        print(json.dumps({"waiting_for_headroom": {"free_pct": f, "need": a.min_free_pct}}), flush=True)
        J.renew(a.key); time.sleep(30)
    wjson(d / "preflight.json", {"free_pct": f, "swap_used_mb": swap_used_mb(), "at": iso(), "lease": row["lease_id"]})
    J.begin(a.key, str(d))                              # fence: persisted before anything is spawned
    spawn_supervisor(a.key, adopt=False)
    print(json.dumps({"begun": True, "execution": str(d), "lease": row["lease_id"]}), flush=True)
    if a.wait:
        return wait_done(a.key)


def spawn_supervisor(key, adopt):
    d = jdir(key)
    env = dict(os.environ, **{SUP_MARK: key})
    env.pop(MARK, None)
    args = [sys.executable, os.path.abspath(__file__), "_supervise", key] + (["--adopt"] if adopt else [])
    out = open(d / "supervisor.log", "a")
    p = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
    wjson(d / "supervisor.json", {"pid": p.pid, "started": iso(), "adopt": adopt})


def cmd_supervise(a):
    key = a.key; d = jdir(key); intent = rjson(d / "intent.json"); J = jobs()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)          # the work is owned; a stray TERM must not orphan bookkeeping
    if not a.adopt:
        if rjson(d / "exec.json"):
            log(key, "exec.json exists: refusing to spawn again"); return 2
        env = dict(os.environ, **{MARK: key}); env.pop(SUP_MARK, None)
        out = open(d / "out.log", "ab")
        # A thin sh owns the workload and records its return code in the job dir, so an adopted run still knows it.
        wrap = ["/bin/sh", "-c", '"$@"; rc=$?; echo $rc > "$WAFER_JOB_DIR/rc.tmp" && mv "$WAFER_JOB_DIR/rc.tmp" "$WAFER_JOB_DIR/rc"; exit $rc', "wafer-job"]
        env["WAFER_JOB_DIR"] = str(d)
        child = subprocess.Popen(wrap + intent["argv"], cwd=intent["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        st = owned(key) or []
        wjson(d / "exec.json", {"pid": child.pid, "pgid": child.pid, "spawned_at": iso(), "start": next((p["start"] for p in st if p["pid"] == child.pid), None)})
        log(key, f"spawned pid {child.pid}")
    else:
        child = None
        log(key, "adopt: reconnecting to the recorded execution; no relaunch")
    ttl = J.required(key)["request"]["ttl_seconds"]; every = max(30, ttl // 3); last_renew = now()
    peak = {"owned_rss_mb": 0.0, "min_free_pct": None, "max_swap_mb": None}
    rc = None
    while True:
        if child is not None:
            rc = child.poll()
        procs = owned(key)
        f = free_pct(); s = swap_used_mb(); rss = sum(p["rss_kb"] for p in procs) / 1024 if procs else 0.0
        with open(d / "samples.jsonl", "a") as fh:
            fh.write(json.dumps({"t": iso(), "free_pct": f, "swap_mb": s, "owned": len(procs) if procs is not None else None, "owned_rss_mb": round(rss, 1)}) + "\n")
        peak["owned_rss_mb"] = max(peak["owned_rss_mb"], rss)
        if f is not None: peak["min_free_pct"] = f if peak["min_free_pct"] is None else min(peak["min_free_pct"], f)
        if s is not None: peak["max_swap_mb"] = s if peak["max_swap_mb"] is None else max(peak["max_swap_mb"], s)
        if now() - last_renew >= every:
            try:
                row = J.renew(key); last_renew = now()
                log(key, f"renewed: state {row['state']} expires {row.get('receipt', {}).get('expires_at')}")
            except Exception as e:                         # hold policy keeps an expired claim charged
                log(key, f"renew failed (claim retained, hold policy): {e!r}"); last_renew = now()
        if procs is not None and not procs and (child is None or rc is not None):
            break
        time.sleep(10)
    if rc is None and child is not None:
        rc = child.wait()
    try:
        rc = int((d / "rc").read_text())
    except (FileNotFoundError, ValueError):
        rc = rc if child is not None else None             # None: killed before the wrapper could record it
    peak = whole_run_peak(d)
    wjson(d / "exit.json", {"returncode": rc, "ended_at": iso(), "peak": peak, "adopted": bool(a.adopt)})
    log(key, f"workload gone: rc {rc}; peak {peak}")
    for attempt in range(1, 6):
        ev = observe(key)
        if ev["state"] != "gone":
            log(key, f"observer says {ev['state']}; keeping the claim"); time.sleep(10); continue
        try:
            row = J.release(key, ev)
            wjson(d / "release.json", {"state": row["state"], "lease": row["lease_id"], "receipt": row.get("receipt"), "evidence": ev, "at": iso()})
            log(key, f"released: {row['state']} lease {row['lease_id']}")
            return 0
        except Exception as e:
            log(key, f"release attempt {attempt} failed: {e!r}"); time.sleep(10)
    log(key, "release not acknowledged: release_pending stays on the record; run `wafer-job attach` to retry")
    return 1


def whole_run_peak(d):
    """Peak over every sample of the run, across supervisor restarts."""
    rows = [json.loads(l) for l in open(d / "samples.jsonl")] if (d / "samples.jsonl").exists() else []
    vals = lambda k: [r[k] for r in rows if r.get(k) is not None]
    return {"owned_rss_mb": max(vals("owned_rss_mb"), default=0.0), "min_free_pct": min(vals("free_pct"), default=None),
            "max_swap_mb": max(vals("swap_mb"), default=None), "samples": len(rows),
            "first": rows[0]["t"] if rows else None, "last": rows[-1]["t"] if rows else None}


def cmd_attach(a):
    d = jdir(a.key); sup = rjson(d / "supervisor.json") or {}
    if owned(a.key, SUP_MARK):
        print(json.dumps({"attach": "supervisor alive", "supervisor_pid": sup.get("pid")})); return 0
    if rjson(d / "release.json"):
        print(json.dumps({"attach": "already released", "release": rjson(d / "release.json")["state"]})); return 0
    row = jobs().refresh(a.key)
    if row["workload"] == "not_started":
        print(json.dumps({"attach": "never begun", "state": row["state"]})); return 0
    spawn_supervisor(a.key, adopt=True)
    print(json.dumps({"attach": "supervisor restarted in adopt mode (no relaunch)", "observe": observe(a.key)})); return 0


def wait_done(key):
    while not rjson(jdir(key) / "release.json"):
        time.sleep(15)
    print(json.dumps(rjson(jdir(key) / "release.json"), default=str)); return 0


def cmd_status(a):
    d = jdir(a.key)
    row = jobs().refresh(a.key)
    print(json.dumps({"claim": {k: row.get(k) for k in ("state", "lease_id", "workload", "execution")},
                      "expires_at": (row.get("receipt") or {}).get("expires_at"),
                      "observe": observe(a.key), "exec": rjson(d / "exec.json"), "exit": rjson(d / "exit.json"),
                      "release": (rjson(d / "release.json") or {}).get("state")}, default=str, indent=1))


def main():
    p = argparse.ArgumentParser(prog="wafer-job", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("key"); r.add_argument("--purpose", required=True)
    m = r.add_mutually_exclusive_group(required=True); m.add_argument("--exclusive", action="store_true"); m.add_argument("--memory", type=int)
    r.add_argument("--service"); r.add_argument("--ttl", type=int, default=1800); r.add_argument("--max-total", type=int, default=21600)
    r.add_argument("--min-free-pct", type=int, default=25, help="deployment headroom policy (system-wide free %%) before begin")
    r.add_argument("--cwd", default="."); r.add_argument("--wait", action="store_true")
    for c in ("attach", "status", "observe"):
        x = sub.add_parser(c); x.add_argument("key")
    s = sub.add_parser("_supervise"); s.add_argument("key"); s.add_argument("--adopt", action="store_true")
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    a = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if a.cmd == "run":
        if not cmd:
            p.error("command required after --")
        a.argv = cmd
        return cmd_run(a)
    if a.cmd == "observe":
        print(json.dumps(observe(a.key))); return 0
    return {"attach": cmd_attach, "status": cmd_status, "_supervise": cmd_supervise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
