#!/usr/bin/env python3
"""Benchmark harness CLI for rl-baselines."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "bench"


class BenchError(RuntimeError):
    pass


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise BenchError(f"Expected mapping in {path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def required_check(data: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = load_yaml(schema_path)
    required = schema.get("required", [])
    missing = [k for k in required if k not in data]
    if missing:
        raise BenchError(f"{label} missing required keys: {', '.join(missing)}")


def resolve(path_like: str) -> Path:
    return (ROOT / path_like).resolve()


def enabled_baselines(exp: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for row in exp["baselines"]:
        if row.get("enabled", True):
            out.append(row["id"])
    return out


def load_adapter(baseline_id: str) -> dict[str, Any]:
    return load_yaml(BENCH / "baselines" / baseline_id / "adapter.yaml")


def load_upstreams() -> dict[str, Any]:
    return load_yaml(BENCH / "tracking" / "upstreams.yaml")


def git_head(repo: Path) -> str:
    return run_cmd(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()


def git_status(repo: Path, paths: Iterable[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    return run_cmd(cmd, cwd=repo).stdout.strip()


def apply_patch_file(repo: Path, patch_file: Path) -> str:
    if not patch_file.exists():
        return "missing"

    forward = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    if forward.returncode == 0:
        run_cmd(["git", "apply", str(patch_file)], cwd=repo, check=True)
        return "applied"

    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch_file)],
        cwd=str(repo),
        text=True,
        capture_output=True,
    )
    if reverse.returncode == 0:
        return "already_applied"

    return "conflict"


def select_zsh_shell() -> str:
    candidates: list[str] = []
    env_shell = os.environ.get("SHELL")
    if env_shell and "zsh" in Path(env_shell).name:
        candidates.append(env_shell)
    candidates.extend(["/bin/zsh", "/usr/bin/zsh"])

    seen: set[str] = set()
    for shell in candidates:
        if shell in seen:
            continue
        seen.add(shell)
        if Path(shell).exists():
            return shell

    raise BenchError("zsh shell not found. Install zsh or set SHELL to a zsh path.")


def detect_command(exe: str) -> tuple[bool, str]:
    path = shutil.which(exe)
    if path:
        return True, f"path:{path}"

    try:
        shell = select_zsh_shell()
    except BenchError:
        return False, "missing"

    probe = subprocess.run(
        [shell, "-lic", f"command -v {shlex.quote(exe)}"],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0 and probe.stdout.strip():
        first = probe.stdout.strip().splitlines()[0]
        return True, f"shell:{first}"

    return False, "missing"


def cmd_doctor(_: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str, bool]] = []
    required_cmds = ["git", "python", "zsh", "micromamba", "sbatch", "curl", "tar"]
    optional_cmds = ["gh"]

    for exe in required_cmds:
        ok, msg = detect_command(exe)
        checks.append((f"command:{exe}", ok, msg, True))

    for exe in optional_cmds:
        ok, msg = detect_command(exe)
        checks.append((f"command:{exe}", ok, msg, False))

    gh_ok = False
    gh_msg = "gh command missing"
    gh_present = dict((name, ok) for name, ok, _, _ in checks).get("command:gh", False)
    if gh_present:
        try:
            shell = select_zsh_shell()
        except BenchError as exc:
            gh_ok = False
            gh_msg = str(exc)
            checks.append(("gh_auth", gh_ok, gh_msg, False))
            for name, ok, msg, required in checks:
                if ok:
                    status = "OK"
                elif required:
                    status = "FAIL"
                else:
                    status = "WARN"
                print(f"[{status}] {name}: {msg}")
            return 0 if all(ok for _, ok, _, required in checks if required) else 1

        proc = subprocess.run([shell, "-lic", "gh auth status"], text=True, capture_output=True)
        gh_ok = proc.returncode == 0
        out = (proc.stdout + proc.stderr).strip()
        if gh_ok:
            gh_msg = "authenticated"
        else:
            gh_msg = out.splitlines()[-1] if out else "gh auth status failed"
    else:
        gh_ok = True
        gh_msg = "skipped (gh optional)"

    checks.append(("gh_auth", gh_ok, gh_msg, False))

    for name, ok, msg, required in checks:
        if ok:
            status = "OK"
        elif required:
            status = "FAIL"
        else:
            status = "WARN"
        print(f"[{status}] {name}: {msg}")

    return 0 if all(ok for _, ok, _, required in checks if required) else 1


def cmd_validate(args: argparse.Namespace) -> int:
    exp_path = resolve(args.experiment)
    exp = load_yaml(exp_path)
    required_check(exp, BENCH / "schemas" / "experiment.schema.yaml", "experiment")

    upstreams = load_upstreams().get("baselines", {})

    errors: list[str] = []
    for baseline_id in enabled_baselines(exp):
        adapter_path = BENCH / "baselines" / baseline_id / "adapter.yaml"
        if not adapter_path.exists():
            errors.append(f"Missing adapter for baseline '{baseline_id}'")
            continue

        adapter = load_yaml(adapter_path)
        try:
            required_check(adapter, BENCH / "schemas" / "adapter.schema.yaml", f"adapter:{baseline_id}")
        except BenchError as exc:
            errors.append(str(exc))
            continue

        repo = resolve(adapter["repo_path"])
        if not repo.exists():
            errors.append(f"{baseline_id}: repo path missing: {repo}")
        else:
            for rel in adapter.get("preflight_checks", {}).get("required_files", []):
                if not (repo / rel).exists():
                    errors.append(f"{baseline_id}: missing required file: {repo / rel}")

        if baseline_id not in exp.get("env_alias_by_baseline", {}):
            errors.append(f"{baseline_id}: missing env_alias_by_baseline entry")

        if baseline_id not in upstreams:
            errors.append(f"{baseline_id}: missing upstream entry")

        if exp["task_id"] not in adapter.get("task_mapping", {}):
            errors.append(f"{baseline_id}: task mapping missing for {exp['task_id']}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print(f"Validation passed: {exp_path}")
    return 0


def bootstrap_one(
    baseline_id: str,
    cfg: dict[str, Any],
    *,
    checkout: bool,
    force: bool,
    apply_patches: bool,
) -> None:
    repo = resolve(cfg["repo_path"])
    remote = cfg["remote_url"]

    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {baseline_id} -> {repo}")
        run_cmd(["git", "clone", remote, str(repo)], cwd=ROOT)

    if not (repo / ".git").exists():
        raise BenchError(f"{baseline_id}: missing git repo at {repo}")

    run_cmd(["git", "fetch", "origin"], cwd=repo)

    if checkout:
        dirty = git_status(repo)
        if dirty and not force:
            raise BenchError(
                f"{baseline_id}: repo is dirty; refusing checkout. "
                f"Commit/stash changes or rerun with --force."
            )
        print(f"checkout {baseline_id} -> {cfg['upstream_commit']}")
        run_cmd(["git", "checkout", cfg["upstream_commit"]], cwd=repo)

    run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=repo)

    if apply_patches:
        for rel_patch in cfg.get("patches", []):
            patch = resolve(rel_patch)
            state = apply_patch_file(repo, patch)
            print(f"patch {baseline_id}: {patch} -> {state}")
            if state in {"missing", "conflict"}:
                raise BenchError(
                    f"{baseline_id}: patch apply failed ({state}) for {patch}. "
                    "Fix conflicts or refresh the patch, then rerun bootstrap."
                )


def create_env_from_spec(baseline_id: str) -> None:
    spec = load_yaml(BENCH / "envs" / f"{baseline_id}.environment.yml")
    commands = spec.get("create", {}).get("commands", [])
    if not commands:
        print(f"WARN {baseline_id}: no env create commands")
        return

    shell = select_zsh_shell()

    for command in commands:
        print(f"env[{baseline_id}] $ {command}")
        proc = subprocess.run(
            [shell, "-lic", command],
            cwd=str(ROOT),
            text=True,
        )
        if proc.returncode != 0:
            raise BenchError(
                f"{baseline_id}: env command failed under shell '{shell}': {command}"
            )


def cmd_bootstrap(args: argparse.Namespace) -> int:
    upstreams = load_upstreams().get("baselines", {})
    selected = [args.baseline] if args.baseline else list(upstreams.keys())

    for baseline_id in selected:
        if baseline_id not in upstreams:
            raise BenchError(f"Unknown baseline: {baseline_id}")
        bootstrap_one(
            baseline_id,
            upstreams[baseline_id],
            checkout=not args.no_checkout,
            force=args.force,
            apply_patches=not args.no_apply_patches,
        )
        if args.create_envs:
            create_env_from_spec(baseline_id)

    print("bootstrap complete")
    return 0


def patch_state(repo: Path, patch: Path) -> str:
    if not patch.exists():
        return "missing"

    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if reverse.returncode == 0:
        return "applied"

    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if forward.returncode == 0:
        return "not_applied"

    return "conflict"


def cmd_tracking_status(_: argparse.Namespace) -> int:
    upstreams = load_upstreams().get("baselines", {})
    any_bad = False

    for baseline_id, cfg in upstreams.items():
        repo = resolve(cfg["repo_path"])
        if not repo.exists():
            print(f"[{baseline_id}] MISSING repo: {repo}")
            any_bad = True
            continue

        head = git_head(repo)
        pinned = cfg["upstream_commit"]
        status = git_status(repo, cfg.get("track_paths", []))

        head_ok = head == pinned
        dirty = bool(status)

        print(f"[{baseline_id}] head={head} pinned={pinned} match={head_ok}")
        if dirty:
            print(f"[{baseline_id}] tracked changes:\n{status}")

        for rel_patch in cfg.get("patches", []):
            patch = resolve(rel_patch)
            state = patch_state(repo, patch)
            print(f"[{baseline_id}] patch {patch}: {state}")
            if state in {"missing", "conflict"}:
                any_bad = True

        if not head_ok:
            any_bad = True
        if dirty:
            any_bad = True

    return 1 if any_bad else 0


def refresh_patch(baseline_id: str, cfg: dict[str, Any], out_file: Path) -> None:
    repo = resolve(cfg["repo_path"])
    paths = cfg.get("track_paths", [])
    if not paths:
        raise BenchError(f"{baseline_id}: no track_paths")

    upstream_ref = cfg["upstream_ref"]

    pieces: list[str] = []

    committed = run_cmd(
        ["git", "format-patch", "--stdout", "--binary", f"{upstream_ref}..HEAD", "--", *paths],
        cwd=repo,
    ).stdout
    if committed.strip():
        pieces.append(committed)

    working = run_cmd(["git", "diff", "--binary", "--", *paths], cwd=repo).stdout
    if working.strip():
        pieces.append(working)

    untracked = run_cmd(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=repo
    ).stdout.splitlines()
    for rel in untracked:
        rel = rel.strip()
        if not rel:
            continue
        proc = run_cmd(
            ["git", "diff", "--binary", "--no-index", "/dev/null", rel],
            cwd=repo,
            check=False,
        )
        if proc.stdout.strip():
            pieces.append(proc.stdout)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(p.strip("\n") for p in pieces if p.strip())
    if content:
        out_file.write_text(content + "\n", encoding="utf-8")
    else:
        out_file.write_text("# empty patch\n", encoding="utf-8")


def cmd_tracking_refresh_patch(args: argparse.Namespace) -> int:
    upstreams = load_upstreams().get("baselines", {})
    baseline_id = args.baseline
    if baseline_id not in upstreams:
        raise BenchError(f"Unknown baseline: {baseline_id}")

    cfg = upstreams[baseline_id]
    patch_list = cfg.get("patches", [])
    if not patch_list:
        raise BenchError(f"{baseline_id}: no patch output path configured")

    out = resolve(patch_list[0])
    refresh_patch(baseline_id, cfg, out)
    print(f"wrote patch: {out}")
    return 0


def cmd_lock_envs(args: argparse.Namespace) -> int:
    baseline_ids: list[str]
    if args.baseline:
        baseline_ids = [args.baseline]
    else:
        baseline_ids = [p.stem.replace(".environment", "") for p in (BENCH / "envs").glob("*.environment.yml")]

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for baseline_id in sorted(set(baseline_ids)):
        spec_path = BENCH / "envs" / f"{baseline_id}.environment.yml"
        if not spec_path.exists():
            raise BenchError(f"Missing env spec: {spec_path}")
        spec_text = spec_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()

        lock = {
            "version": 1,
            "generated": now,
            "platform": "linux-64",
            "baseline": baseline_id,
            "spec_sha256": digest,
            "spec_file": str(spec_path.relative_to(ROOT)),
        }
        lock_path = BENCH / "envs" / "locks" / f"{baseline_id}.linux-64.lock.yml"
        save_yaml(lock_path, lock)
        print(f"wrote lock: {lock_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rl-baselines harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Check host/tooling health")
    p_doctor.set_defaults(func=cmd_doctor)

    p_validate = sub.add_parser("validate", help="Validate experiment and adapters")
    p_validate.add_argument("--experiment", required=True, help="Path to experiment yaml")
    p_validate.set_defaults(func=cmd_validate)

    p_bootstrap = sub.add_parser("bootstrap", help="Clone/sync/apply patches for baselines")
    p_bootstrap.add_argument("--baseline", help="Bootstrap one baseline")
    p_bootstrap.add_argument("--no-checkout", action="store_true", help="Skip checkout to pinned commit")
    p_bootstrap.add_argument("--force", action="store_true", help="Allow checkout even when repo is dirty")
    p_bootstrap.add_argument("--no-apply-patches", action="store_true", help="Skip patch apply")
    p_bootstrap.add_argument("--create-envs", action="store_true", help="Execute env creation commands")
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_tracking = sub.add_parser("tracking", help="Tracking operations")
    tracking_sub = p_tracking.add_subparsers(dest="tracking_command", required=True)

    p_tracking_status = tracking_sub.add_parser("status", help="Show tracking state")
    p_tracking_status.set_defaults(func=cmd_tracking_status)

    p_tracking_refresh = tracking_sub.add_parser(
        "refresh-patch", help="Refresh patch from tracked paths"
    )
    p_tracking_refresh.add_argument("--baseline", required=True, help="Baseline id")
    p_tracking_refresh.set_defaults(func=cmd_tracking_refresh_patch)

    p_lock = sub.add_parser("lock-envs", help="Update lock metadata from env specs")
    p_lock.add_argument("--baseline", help="Generate lock for one baseline")
    p_lock.set_defaults(func=cmd_lock_envs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
