# Pinned 235B checkpoint staging

The three-host small-model gate is complete before this operator is packaged.
This stages the selected Qwen3-235B-A22B-Instruct-2507 6-bit revision on Obelisk's
local disk. It imports only the standard library and changes no service Python
packages, gateway configuration, lease, model process or inference state.

The plan pins the repository revision, all 37 LFS weight hashes and file sizes,
metadata Git blob or LFS checksums, destination, service UID and the completed
three-host qualification proof. `package.py` creates a fresh packet, substitutes
the exact remote path and plan SHA, and prints both source hashes for review.
It does not copy the packet, launch a download or grant privileged access.

```sh
python3 ops/exo-checkpoint/package.py --plan reviewed-plan.json \
  --remote-packet /Users/obelisk-services/var/obelisk-services/exo-qualification/qwen235b-download-v1 \
  --destination /path/to/fresh-packet
```

After verifying and copying the packet as the Obelisk service account, use its
existing interpreter with `-I -B` and `download.py --start`. A durable supervised
job or an attended operator should retain stdout and the process identity.
Do not repeat an unknown launch: inspect that exact process and destination.
The real rollout records that launch once and runs it detached at nice 10.

The downloader requires enough measured free disk space for the full remaining
inventory plus 8 GiB. It uses no Hugging Face token or home cache. Data goes into
an exclusively owned staging directory. Each file must match its full pinned
checksum before atomic publication by a non-overwriting hard link. Final files
and the completed tree are read-only. `COMPLETE.json` records every full SHA256.
A crash during publication stays unqualified; an ambiguous extra hard link is
refused, not silently repaired.

A failed transfer preserves its partial bytes. An explicit
`--continue-download` validates the same plan and resumes only a correctly
honored HTTP range, then verifies the entire file. Completed files are verified
and reused. A checksum error, ignored range, symlink, unsafe name, short or
oversized response prevents publication. This is a data transfer continuation,
not an inference retry or permission to replay a compute request.

A complete checkpoint is storage evidence. Assigned-rank artifacts, actual
resident fit, tool calling and the evaluation suite remain separate gates.

```sh
python3 -m unittest discover -s ops/exo-checkpoint -p 'test_*.py' -v
```

Eight byte-stream controls and three packet controls run without network or models.
The packet controls reject incomplete qualification and path/code injection before
creating any output. The packager reproduces both live packet files byte for byte. A real rehearsal
verified all ten pinned metadata files (16,117,761 bytes). The live stager began
only after the original three prompts matched all 16 raw IDs on all three ranks,
all production ends were confirmed and every lease was exactly settled.
