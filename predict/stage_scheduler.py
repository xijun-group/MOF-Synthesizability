"""Run independent descriptor stages in isolated, bounded child processes.

The numerical implementations remain in predict_synthesizability.py. Processes,
rather than threads executing those functions, isolate sys.argv, module globals,
MOFid imports, and the signal handlers used by molSimplify.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback


def _write(path, doc):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    tmp.replace(path)


def _worker(task_file):
    task = json.loads(Path(task_file).read_text())
    spec = importlib.util.spec_from_file_location("scheduled_predict", task["driver"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    ctx = task["ctx"]
    for key in ("prefix", "model_dir", "predict_dir", "rules"):
        if ctx.get(key) is not None:
            ctx[key] = Path(ctx[key])
    cif, work = Path(task["cif"]), Path(task["work"])
    stage, args = task["stage"], task["args"]
    started = time.monotonic()
    try:
        if stage == "channel_b":
            result = mod.step_channel_b(ctx, task["identities"], work,
                                       args["workers"], args["timeout"])
        elif stage == "rmsd":
            sa_scores = mod.score_fragments(ctx, task["identities"])
            result = mod.step_rmsd(ctx, cif, work, args["timeout"], args["rmsd_supercell"],
                                   sa_scores=sa_scores)
        elif stage == "mofsimplify":
            result = mod.step_mofsimplify(ctx, cif, task["label"], work, args["timeout"])
        elif stage == "ehull":
            result = mod.step_ehull(ctx, cif, task["label"], work)
        else:
            raise ValueError("unknown descriptor stage " + stage)
        _write(task["result"], {"state": "complete", "stage": stage,
                               "wall_s": time.monotonic() - started, "result": result})
        return 0
    except BaseException as exc:
        _write(task["result"], {"state": "failed", "stage": stage,
                               "wall_s": time.monotonic() - started,
                               "error": repr(exc), "traceback": traceback.format_exc()})
        return 1


def collect(ctx, args, cif, label, work, identities, skipped):
    """Return the original stage dictionaries; preserve missing-feature behavior.

    ``identities`` is the standardised fragment set, or a callable returning it. With a
    callable the stages that do not read it start at once; RMSD/channel-B resolve it when they
    launches, and a failure of the callable cancels every running stage."""
    stages = {
        # Start GPU/model work first; it is the critical path on the measured
        # reference. Short CPU work fills the remaining slots.
        "ehull": "ehull" not in skipped,
        "rmsd": "rmsd" not in skipped,
        "mofsimplify": not ({"mofsimplify", "solvent_removal_stability"} & skipped),
        "channel_b": not ({"b", "channel_b", "linker_homo"} & skipped),
    }
    if stages["ehull"] and (not ctx["ehull_python"] or not Path(ctx["ehull_python"]).exists()):
        raise RuntimeError("Ehull interpreter missing; pass --ehull-python or --skip ehull")
    journal = work / "stage_logs"
    journal.mkdir(exist_ok=True)
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("parallel stages require bwrap for descendant process cleanup")
    running = {}
    lock = threading.Lock()
    stopping = threading.Event()
    started = time.monotonic()
    results = {stage: {"skipped": True} for stage, enabled in stages.items() if not enabled}
    records = {}

    def terminate_group(proc):
        # A leader can exit before its descendants. Do not use leader.poll() as
        # evidence that its process group is empty.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def one(stage):
        task_path = journal / (stage + ".task.json")
        result_path = journal / (stage + ".result.json")
        result_path.unlink(missing_ok=True)
        task = {"driver": str(ctx["predict_dir"] / "predict_synthesizability.py"),
                "ctx": ctx, "cif": cif, "work": work, "label": label,
                "identities": ((identities() if callable(identities) else identities)
                               if stage in ("channel_b", "rmsd") else None),
                "stage": stage, "result": result_path,
                "args": {k: getattr(args, k) for k in ("workers", "timeout", "rmsd_supercell")}}
        _write(task_path, task)
        launch = time.monotonic()
        with (journal / (stage + ".log")).open("w") as output:
            with lock:
                if stopping.is_set():
                    raise RuntimeError("descriptor stage cancelled before launch")
                # A PID namespace also owns descendants that start new process
                # groups (notably GNU timeout). Its init exit kills all of them.
                command = [bwrap, "--die-with-parent", "--unshare-pid", "--bind", "/", "/",
                           "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--",
                           sys.executable, str(Path(__file__).resolve()), str(task_path)]
                proc = subprocess.Popen(command,
                                        stdout=output, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[stage] = proc
            try:
                rc = proc.wait()
            finally:
                terminate_group(proc)
                with lock:
                    running.pop(stage, None)
        rec = json.loads(result_path.read_text()) if result_path.exists() else {}
        rec.update(returncode=rc, process_wall_s=time.monotonic() - launch,
                   log=str(journal / (stage + ".log")))
        if rc or rec.get("state") != "complete":
            raise RuntimeError(f"{stage} failed; see {journal / (stage + '.log')}: {rec.get('error', rc)}")
        return rec

    def interrupted(signum, frame):
        raise InterruptedError(f"descriptor scheduler received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.stage_jobs)
    futures = {pool.submit(one, stage): stage for stage, enabled in stages.items() if enabled}
    if callable(identities):
        # Surfaces a failed fragment set while every pool slot is still busy. A daemon
        # thread, so an abandoned wait does not hold the interpreter at exit.
        gate = concurrent.futures.Future()

        def watch():
            try:
                gate.set_result(identities())
            except BaseException as exc:
                gate.set_exception(exc)

        threading.Thread(target=watch, daemon=True).start()
        futures[gate] = None
    try:
        for future in concurrent.futures.as_completed(futures):
            stage = futures[future]
            if stage is None:
                future.result()
                continue
            records[stage] = future.result()
            results[stage] = records[stage].pop("result")
    except BaseException:
        stopping.set()
        for future in futures:
            future.cancel()
        # Every running group was created here; no pre-existing jobs are touched.
        with lock:
            owned = list(running.values())
        for proc in owned:
            terminate_group(proc)
        _write(journal / "receipt.json", {"state": "failed", "stage_jobs": args.stage_jobs,
                                          "wall_s": time.monotonic() - started, "stages": records})
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
    _write(journal / "receipt.json", {"state": "complete", "stage_jobs": args.stage_jobs,
                                      "wall_s": time.monotonic() - started, "stages": records})
    return results


if __name__ == "__main__":
    raise SystemExit(_worker(sys.argv[1]))
