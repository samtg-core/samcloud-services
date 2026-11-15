#!/usr/bin/env python3
"""Metal available memory credits clean file cache, but never a resident model (task 978).

On Obelisk on 8 Nov 2026 the collector summed free + inactive + speculative pages
(110,777 MiB) and the gateway refused qwen3.5-122b:q6_k's 108,544 MiB cold load
for a day: at 95% it needs 114,258. vm_stat said 106.8 GiB was file-backed and
9.2 GiB anonymous; active file cache was being counted as used. The collector now
takes the larger of the old sum and free + file-backed - accelerator in-use.

Runs anywhere: vm_stat, sysctl and ioreg are stubbed with recorded output.

    python -m ollama.test_capacity_metal_reclaimable
"""
import subprocess
import sys

from . import capacity

# Recorded from obelisk-services, 2026-11-08 21:10 UTC, nothing loaded.
OBELISK_20261108 = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   510002.
Pages active:                                1022664.
Pages inactive:                              6499603.
Pages speculative:                             80109.
Pages throttled:                                   0.
Pages wired down:                             191080.
Pages purgeable:                               25614.
"Translation faults":                     1174231034.
Pages copy-on-write:                        37174459.
Pages zero filled:                        1507563486.
Pages reactivated:                          19405111.
Pages purged:                                2726597.
File-backed pages:                           6996988.
Anonymous pages:                              605388.
Pages stored in compressor:                    78838.
Pages occupied by compressor:                  33309.
Decompressions:                             25467800.
Compressions:                               32209719.
Pageins:                                   387668217.
Pageouts:                                       4274.
Swapins:                                     2263032.
Swapouts:                                    2467131.
Pages tagged:                                 139199.
Pages tagged resident:                        121535.
Pages tagged compressed:                       17664.
Pages tag-storage:                            262144.
Pages tag-storage holding tags:                 5165.
Pages tag-storage free:                        11946.
Pages tag-storage non-tag pageable:           241301.
Pages tag-storage non-tag wired:                   8.
Bytes of compressed tags:                    2599232.
Tagged compressions:                          307025.
Tagged decompressions:                        270783.
"""
IDLE_INUSE_MB = 17          # ioreg "In use system memory", same minute

# The same box without the two noun-last counters (older macOS output).
NO_FILE_BACKED = "\n".join(l for l in OBELISK_20261108.splitlines()
                           if not l.startswith(("File-backed", "Anonymous")))


def vm(free, active, inactive, speculative, wired, file_backed, anonymous, compressor):
    return f"""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               {free}.
Pages active:                             {active}.
Pages inactive:                           {inactive}.
Pages speculative:                        {speculative}.
Pages wired down:                         {wired}.
File-backed pages:                        {file_backed}.
Anonymous pages:                          {anonymous}.
Pages occupied by compressor:             {compressor}.
"""

GIB = 65536                 # 16 KiB pages per GiB

# (b) A 96 GiB model resident through mmap: its weights are hot file-backed
# pages in ACTIVE, and the accelerator counts them in use. 8 GiB of other
# cache is inactive. Only free + the other cache may count.
MMAP_RESIDENT = vm(free=4 * GIB, active=97 * GIB, inactive=8 * GIB, speculative=0,
                   wired=3 * GIB, file_backed=104 * GIB, anonymous=1 * GIB, compressor=0)
MMAP_INUSE_MB = 96 * 1024

# (c) Real pressure: anonymous memory and the compressor are large, file cache small.
PRESSURE = vm(free=2 * GIB, active=70 * GIB, inactive=20 * GIB, speculative=1 * GIB,
              wired=6 * GIB, file_backed=6 * GIB, anonymous=85 * GIB, compressor=25 * GIB)

failures, checks = [], 0


def check(cond, msg):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def reading(vm_stat_out, inuse_mb):
    real_run, accel, load = subprocess.run, capacity._accel_stats, capacity._load_avg

    def fake_run(args, **kw):
        out = vm_stat_out if args[0] == capacity._VM_STAT else (
            str(128 * 1024 ** 3) if args[0] == capacity._SYSCTL else '')
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr='')
    capacity.subprocess.run = fake_run
    capacity._accel_stats = lambda: (0.0, inuse_mb)
    capacity._load_avg = lambda: 0.1
    try:
        return capacity._collect_metal()
    finally:
        capacity.subprocess.run, capacity._accel_stats, capacity._load_avg = real_run, accel, load


def mb(pages):
    return round(pages * 16384 / 1024 / 1024)


def fits_at(need, available, fraction, headroom):
    f, h = capacity.USABLE_FRACTION, capacity.MIN_HEADROOM_MB
    capacity.USABLE_FRACTION, capacity.MIN_HEADROOM_MB = fraction, headroom
    try:
        return capacity.fits(need, available), capacity.usable_mb(available)
    finally:
        capacity.USABLE_FRACTION, capacity.MIN_HEADROOM_MB = f, h


def main():
    old = mb(510002 + 6499603 + 80109)
    print("(a) cold host, large clean file cache")
    r = reading(OBELISK_20261108, IDLE_INUSE_MB)
    want = mb(510002 + 6996988) - IDLE_INUSE_MB
    check(r["memory_available_mb"] == want,
          f"Obelisk 8 Nov: available = free + file-backed - in-use = {want} ({r['memory_available_mb']})")
    ok, usable = fits_at(108544, r["memory_available_mb"], 0.95, 1024)
    check(ok, f"q6_k 108,544 MiB fits at Obelisk's 0.95 / 1024 (usable {usable})")
    ok, usable = fits_at(108544, old, 0.95, 1024)
    check(not ok, f"the old sum {old} is what refused it (usable {usable})")
    check(r["memory_used_mb"] == mb(1022664 + 191080 + 33309), "used is unchanged: active + wired + compressor")
    check(r["memory_available_mb"] <= r["memory_total_mb"], "available never exceeds total")

    print("(b) a model resident through mmap is not counted as free")
    r = reading(MMAP_RESIDENT, MMAP_INUSE_MB)
    old_b = mb(4 * GIB + 8 * GIB)
    check(r["memory_available_mb"] == max(old_b, mb(108 * GIB) - MMAP_INUSE_MB),
          f"available {r['memory_available_mb']} = max(old {old_b}, free + file-backed - in-use)")
    check(r["memory_available_mb"] <= mb(12 * GIB),
          "no more than free + the other cache (12 GiB); the 96 GiB of weights stay used")
    ok, _ = fits_at(96 * 1024, r["memory_available_mb"], 0.95, 1024)
    check(not ok, "a second 96 GiB model is refused")

    print("(c) real memory pressure refuses")
    r = reading(PRESSURE, 2048)
    check(r["memory_available_mb"] == mb(2 * GIB + 20 * GIB + 1 * GIB),
          f"credit never lowers the old sum: available {r['memory_available_mb']} = free + inactive + speculative")
    ok, _ = fits_at(108544, r["memory_available_mb"], 0.95, 1024)
    check(not ok, "q6_k is refused under pressure")

    print("(d) the fraction and headroom still apply")
    a = reading(OBELISK_20261108, IDLE_INUSE_MB)["memory_available_mb"]
    ok, usable = fits_at(108544, a, 0.90, 1024)
    check(not ok and usable == min(int(a * 0.90), a - 1024),
          f"at 0.90 the same reading refuses q6_k (usable {usable})")
    ok, usable = fits_at(108544, a, 0.95, 10000)
    check(not ok and usable == a - 10000, f"a 10,000 MiB headroom wins over the fraction (usable {usable})")

    print("fallbacks")
    r = reading(NO_FILE_BACKED, IDLE_INUSE_MB)
    check(r["memory_available_mb"] == old, f"no file-backed counter: the old sum stands ({r['memory_available_mb']})")
    r = reading(OBELISK_20261108, None)
    check(r["memory_available_mb"] == old, f"in-use unknown: the old sum stands ({r['memory_available_mb']})")
    pages, size = None, None
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
