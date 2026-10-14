"""Report git state of every repo/package HPC points at.

Compares:
- local git state for qsp-hpc-tools, qsp-submodel, maple, qspio-pdac, pdac-paper
- remote git state for the configured simulator checkout on HPC
- installed package versions in the HPC venv

Flags discrepancies. No SSH credentials are read directly; the HPCJobManager
library handles config loading and transport.

Exit 0 clean, 1 on any FAIL-level discrepancy. Prints warnings for drift that
is auto-resolved by the next workflow run (e.g., qsp-hpc-tools pip upgrade).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
PROJECTS = HOME / "Projects"

# Local checkouts we care about. Override via env var for alternate layouts.
LOCAL_REPOS = {
    # SPQSP_PDAC is gone, and was never here: the port into cpp/ completed, the
    # simulator builds from this repo, and ~/Projects/SPQSP_PDAC exists on no
    # machine we use. Its row reported "missing" on every run.
    "qsp-hpc-tools":  Path(os.environ.get("QSP_HPC_TOOLS", PROJECTS / "qsp-hpc-tools")),
    "qsp-submodel":   Path(os.environ.get("QSP_SUBMODEL", PROJECTS / "qsp-submodel")),
    "maple":          Path(os.environ.get("MAPLE", PROJECTS / "maple")),
    "qspio-pdac":     Path(__file__).resolve().parent.parent,
    "pdac-paper":     Path(os.environ.get("PAPER_ROOT",
                                          Path(__file__).resolve().parents[2] / "pdac-paper")),
}


def git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
        return (out.stdout + out.stderr).strip()
    except FileNotFoundError:
        return "(git not found)"


def local_state(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=path)
    sha = git("rev-parse", "--short", "HEAD", cwd=path)
    dirty = bool(git("status", "--porcelain", cwd=path).strip())
    unpushed = git("log", "--oneline", "@{u}..HEAD", cwd=path) if "@" not in branch else ""
    return {
        "exists": True,
        "branch": branch,
        "sha": sha,
        "dirty": dirty,
        "unpushed": bool(unpushed.strip()) if unpushed else False,
    }


def hpc_remote_state(cfg) -> dict:
    """One SSH call: fetch the simulator checkout's branch/SHA and pip versions."""
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager  # noqa: E402
    mgr = HPCJobManager(config=cfg, verbose=False)
    cpp_repo = cfg.cpp_repo_path or ""
    if not cpp_repo and cfg.cpp_binary_path:
        parts = Path(cfg.cpp_binary_path).parents
        if len(parts) >= 5:
            cpp_repo = str(parts[4])
    venv_py = f"{cfg.hpc_venv_path}/bin/python"
    cmd = f"""
set +e
if [ -n "{cpp_repo}" ] && [ -d "{cpp_repo}/.git" ]; then
  echo "CPP_BRANCH:$(git -C '{cpp_repo}' rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "CPP_SHA:$(git -C '{cpp_repo}' rev-parse --short HEAD 2>/dev/null)"
  echo "CPP_DIRTY:$(git -C '{cpp_repo}' status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
fi
if [ -x "{venv_py}" ]; then
  # importlib.metadata is stdlib and works regardless of whether pip/uv
  # are exposed in the venv — PEP 566. Distribution names use hyphens.
  {venv_py} -c "
import importlib.metadata as im
for dist in ('qsp-hpc-tools','qsp-inference','maple'):
    try: v = im.version(dist)
    except Exception: v = ''
    print(f'PIP:{{dist}}={{v}}')
" 2>/dev/null
fi
"""
    status, output = mgr.transport.exec(cmd, timeout=60)
    result = {"ssh_ok": status == 0, "raw": output}
    for line in output.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip()
    return result


def main() -> int:
    try:
        from qsp_hpc.batch.hpc_job_manager import HPCJobManager
    except ImportError:
        print("qsp-hpc-tools not importable; is the local venv active?")
        return 1

    # Load config via the library (no direct credentials read).
    mgr = HPCJobManager(verbose=False)
    cfg = mgr.config

    print("=" * 72)
    print("Configured HPC pointers (from credentials.yaml)")
    print("=" * 72)
    print(f"  cpp.branch        : {cfg.cpp_branch}")
    print(f"  cpp.repo_path     : {cfg.cpp_repo_path or '(derived)'}")
    print(f"  cpp.binary_path   : {cfg.cpp_binary_path or '(unset)'}")
    print(f"  hpc_venv_path     : {cfg.hpc_venv_path}")
    print(f"  qsp-hpc-tools src : {cfg.qsp_hpc_tools_source}")

    print()
    print("=" * 72)
    print("Local checkouts")
    print("=" * 72)
    local = {name: local_state(p) for name, p in LOCAL_REPOS.items()}
    for name, state in local.items():
        if not state["exists"]:
            print(f"  {name:16s} : (not found at {LOCAL_REPOS[name]})")
            continue
        flags = []
        if state["dirty"]:
            flags.append("dirty")
        if state["unpushed"]:
            flags.append("unpushed commits")
        tag = ("  ⚠ " + ", ".join(flags)) if flags else ""
        print(f"  {name:16s} : {state['branch']} @ {state['sha']}{tag}")

    print()
    print("=" * 72)
    print("HPC remote state (single SSH call)")
    print("=" * 72)
    try:
        remote = hpc_remote_state(cfg)
    except Exception as exc:
        print(f"  SSH failed: {exc}")
        return 1

    if not remote.get("ssh_ok"):
        print("  SSH command returned non-zero. Raw output:")
        print(remote.get("raw", ""))
        return 1

    cpp_branch = remote.get("CPP_BRANCH", "(unknown)")
    cpp_sha = remote.get("CPP_SHA", "(unknown)")
    cpp_dirty = remote.get("CPP_DIRTY", "0")
    print(f"  qspio-pdac @ HPC : {cpp_branch} @ {cpp_sha}"
          + (f"  ⚠ dirty ({cpp_dirty} files)" if cpp_dirty not in ("0", "") else ""))
    for pkg in ("qsp-hpc-tools", "qsp-inference", "maple"):
        v = remote.get(f"PIP:{pkg}", "(not installed)")
        # Normalize keys: the line "PIP:qsp-hpc-tools=0.4.2" yielded key
        # "PIP" with the rest merged in raw. Re-parse:
        # Easier: re-extract inline.
    # Re-parse pip lines directly.
    pkg_versions = {}
    for line in remote.get("raw", "").splitlines():
        if line.startswith("PIP:"):
            name_ver = line[4:]
            if "=" in name_ver:
                n, v = name_ver.split("=", 1)
                pkg_versions[n] = v.strip()
    for pkg in ("qsp-hpc-tools", "qsp-inference", "maple"):
        v = pkg_versions.get(pkg, "").strip() or "(not installed)"
        print(f"  {pkg:16s} @ HPC : {v}")

    print()
    print("=" * 72)
    print("Discrepancy check")
    print("=" * 72)
    problems: list[str] = []
    warns: list[str] = []

    if cpp_branch and cpp_branch != cfg.cpp_branch:
        problems.append(
            f"The simulator checkout on HPC is on '{cpp_branch}', credentials "
            f"configure '{cfg.cpp_branch}'. "
            "ensure_cpp_binary() will checkout+pull the configured branch before the next "
            "submit, overwriting any local HPC work."
        )
    if cpp_dirty not in ("0", ""):
        problems.append(
            f"The simulator checkout on HPC is dirty ({cpp_dirty} files). "
            "ensure_cpp_binary() will refuse to fast-forward."
        )

    # cpp.repo_path IS this repo on the cluster, so the local side of the
    # comparison is this checkout -- not a separate SPQSP one, which is why the
    # old row could never disagree with anything.
    model_local = local.get("qspio-pdac") or {}
    if model_local.get("exists") and cpp_sha and model_local.get("sha") != cpp_sha:
        warns.append(
            f"qspio-pdac local SHA {model_local['sha']} vs HPC SHA {cpp_sha}. "
            "Run `git push` here if you have local commits, then the next submit "
            "will auto-fast-forward on HPC."
        )

    if not problems and not warns:
        print("  OK — HPC and local are consistent.")
    for w in warns:
        print(f"  WARN : {w}")
    for p in problems:
        print(f"  FAIL : {p}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
