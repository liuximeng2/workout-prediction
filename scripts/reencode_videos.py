import sys
sys.stdout.reconfigure(encoding='utf-8')

"""Re-encode workout videos to a lower resolution and frame rate.

Walks every ``data_roots`` directory defined in BaseConfig, re-encodes each
video with ffmpeg, and writes the result to a parallel output tree.
Preserves the ``<root>/<class>/<video>`` structure so all other scripts
(precompute_flow, train_two_stream, …) work without modification — just
point ``data_roots`` at the new output directory.

Usage:
    # Defaults: short-side 256 px, 15 fps, copy audio-less, H.264 CRF 23
    python scripts/reencode_videos.py

    # Custom resolution / FPS
    python scripts/reencode_videos.py --size 224 --fps 24 --crf 20

    # Dry-run (print commands, don't execute)
    python scripts/reencode_videos.py --dry_run

    # Re-encode specific input trees
    python scripts/reencode_videos.py \\
        --input_dirs data/verified_data/verified_data/data_btc_10s \\
                     data/verified_data/verified_data/data_crawl_10s \\
                     data/test/test \\
        --output_root data/reencoded

    # After a run with failures: retry only logged failures (overwrites partial outputs)
    python scripts/reencode_videos.py --retry-failed --output_root data/reencoded

    # Re-encode everything, even if output already exists
    python scripts/reencode_videos.py --force --output_root data/reencoded

    # Retry failures: default is strict decode, then an automatic tolerant retry
    python scripts/reencode_videos.py --retry-failed --output_root data/reencoded \\
        --input_dirs ...

    # Skip the strict pass (only tolerant decode) when every file is known-troublesome
    python scripts/reencode_videos.py --retry-failed --recover --output_root data/reencoded \\
        --input_dirs ...

Output layout (mirrors input under --output_root):
    data/reencoded/data_btc_10s/<class>/<video>.mp4
    data/reencoded/data_crawl_10s/<class>/<video>.mp4
    data/reencoded/test/<class>/<video>.mp4
"""

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.base_config import BaseConfig
from utils.dataset import resolve_data_roots

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _collect_videos(roots: list[Path]) -> list[Path]:
    videos = []
    for root in roots:
        if not root.is_dir():
            print(f"[warn] skipping missing root: {root}")
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(p)
    return videos


def _failed_log_path(output_root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (output_root / ".reencode_failed.txt").resolve()


def _load_retry_sources(failed_log: Path, input_roots: list[Path]) -> list[Path]:
    """Paths from the failed log that still exist and lie under one of input_roots."""
    if not failed_log.is_file():
        print(f"error: failed log not found: {failed_log}", file=sys.stderr)
        sys.exit(1)
    raw = [ln.strip() for ln in failed_log.read_text().splitlines() if ln.strip()]
    seen: set[Path] = set()
    out: list[Path] = []
    for line in raw:
        p = Path(line).expanduser().resolve()
        if not p.is_file():
            print(f"[warn] skip missing (retry list): {p}")
            continue
        if not any(p == r or p.is_relative_to(r) for r in input_roots):
            print(f"[warn] skip not under --input_dirs: {p}")
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _build_output_path(video: Path, input_root: Path, output_root: Path) -> Path:
    """Mirror the video's path under output_root, relative to input_root's parent.

    Example:
        video       = data/verified_data/verified_data/data_btc_10s/squat/foo.mp4
        input_root  = data/verified_data/verified_data/data_btc_10s
        output_root = data/reencoded
        →            data/reencoded/data_btc_10s/squat/foo.mp4
    """
    try:
        rel = video.relative_to(input_root)
    except ValueError:
        rel = Path(video.parent.name) / video.name
    return output_root / input_root.name / rel


def _resolve_ffmpeg(explicit: str | None) -> str:
    """Return absolute path to ffmpeg, or exit with install hints."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return str(p.resolve())
        found = shutil.which(explicit)
        if found:
            return found
        print(f"error: --ffmpeg not found: {explicit}", file=sys.stderr)
        sys.exit(1)
    found = shutil.which("ffmpeg")
    if found:
        return found
    print(
        "error: ffmpeg not found on PATH.\n"
        "  macOS:  brew install ffmpeg\n"
        "  Or pass the binary explicitly:  --ffmpeg /opt/homebrew/bin/ffmpeg",
        file=sys.stderr,
    )
    sys.exit(1)


def _ffmpeg_cmd(
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    size: int,
    fps: int,
    crf: int,
    *,
    recover: bool,
    hwaccel_videotoolbox: bool,
) -> list[str]:
    """Build the ffmpeg command for one video.

    ``recover`` adds demux/decode tolerance for mildly damaged files (must still
    open as a known container). It cannot fix a missing ``moov`` / truncated MP4.
    """
    # scale so the shorter side == `size`, keep both dimensions divisible by 2
    vf = f"scale='if(gt(iw,ih),trunc(iw*{size}/ih/2)*2,{size})':'if(gt(iw,ih),{size},trunc(ih*{size}/iw/2)*2)'"
    pre_input: list[str] = []
    if recover:
        pre_input += [
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+discardcorrupt+igndts",
        ]
    if hwaccel_videotoolbox:
        pre_input += ["-hwaccel", "videotoolbox"]
    return [
        ffmpeg_bin,
        "-y",
        *pre_input,
        "-i", str(src),
        "-vf", vf,
        "-r", str(fps),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        "-an",
        "-movflags", "+faststart",
        str(dst),
    ]


def reencode(
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    size: int,
    fps: int,
    crf: int,
    dry_run: bool,
    skip_if_exists: bool,
    recover: bool,
    hwaccel_videotoolbox: bool,
) -> tuple[Path, bool, str]:
    """Re-encode one video.  Returns (src, success, message)."""
    if skip_if_exists and dst.exists():
        return src, True, "skipped (already exists)"

    dst.parent.mkdir(parents=True, exist_ok=True)

    def run_cmd(recover_flag: bool, vt_flag: bool) -> tuple[int, str]:
        cmd = _ffmpeg_cmd(
            ffmpeg_bin, src, dst, size, fps, crf,
            recover=recover_flag, hwaccel_videotoolbox=vt_flag,
        )
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return r.returncode, r.stderr.decode(errors="replace")

    if dry_run:
        cmd = _ffmpeg_cmd(
            ffmpeg_bin, src, dst, size, fps, crf,
            recover=recover,
            hwaccel_videotoolbox=hwaccel_videotoolbox,
        )
        return src, True, "dry-run: " + " ".join(cmd)

    if recover:
        attempts: list[tuple[bool, bool]] = [(True, hwaccel_videotoolbox)]
    else:
        attempts = [(False, hwaccel_videotoolbox), (True, hwaccel_videotoolbox)]

    last_stderr = ""
    for i, (rec, vt) in enumerate(attempts):
        code, last_stderr = run_cmd(rec, vt)
        if code == 0:
            note = " (tolerant decode)" if i > 0 else ""
            return src, True, f"ok → {dst}{note}"

    err_lines = last_stderr.strip().splitlines()[-3:]
    return src, False, "FAILED: " + " | ".join(err_lines)


def main() -> None:
    cfg = BaseConfig()
    repo_root = Path(__file__).resolve().parent.parent

    default_input_dirs = [
        str(p) for p in resolve_data_roots(cfg.data_roots, repo_root=repo_root)
    ]
    # Also include the test root if it exists
    if cfg.test_data_roots:
        test_roots = resolve_data_roots(cfg.test_data_roots, repo_root=repo_root)
        for tr in test_roots:
            if tr.is_dir():
                default_input_dirs.append(str(tr))

    parser = argparse.ArgumentParser(description="Re-encode videos to lower res/FPS.")
    parser.add_argument(
        "--input_dirs", nargs="+", default=default_input_dirs,
        help="Source directory trees (default: data_roots + test_data_roots from config).",
    )
    parser.add_argument(
        "--output_root", type=str, default=str(repo_root / "data" / "reencoded"),
        help="Root directory for re-encoded output (default: data/reencoded).",
    )
    parser.add_argument("--size",    type=int, default=256,
                        help="Short-side pixel size (default: 256).")
    parser.add_argument("--fps",     type=int, default=15,
                        help="Output frame rate (default: 15).")
    parser.add_argument("--crf",     type=int, default=23,
                        help="H.264 CRF quality (0=lossless, 51=worst; default: 23).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel ffmpeg processes (default: 8).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print ffmpeg commands without executing them.")
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Path to ffmpeg binary (default: look up 'ffmpeg' on PATH).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even when the output file already exists.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only re-encode sources listed in the failed log (see --failed-log). Implies overwrite.",
    )
    parser.add_argument(
        "--failed-log",
        type=str,
        default=None,
        help="Path for failed-source list (default: <output_root>/.reencode_failed.txt).",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Use tolerant demux/decode only (skip the strict first pass). "
             "Default without this flag: strict pass, then automatic tolerant retry on failure.",
    )
    parser.add_argument(
        "--hwaccel-videotoolbox",
        action="store_true",
        help="Use VideoToolbox hwaccel before decode (macOS; different path, sometimes helps).",
    )
    args = parser.parse_args()

    if args.dry_run and args.ffmpeg is None:
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    else:
        ffmpeg_bin = _resolve_ffmpeg(args.ffmpeg)

    input_roots = [Path(d).resolve() for d in args.input_dirs]
    output_root = Path(args.output_root).resolve()
    failed_log = _failed_log_path(output_root, args.failed_log)
    skip_if_exists = not args.force and not args.retry_failed

    print(f"Input roots  : {[str(r) for r in input_roots]}")
    print(f"Output root  : {output_root}")
    print(f"Resolution   : short-side {args.size}px")
    print(f"Frame rate   : {args.fps} fps")
    print(f"CRF          : {args.crf}")
    print(f"Workers      : {args.workers}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Skip existing: {skip_if_exists}")
    print(f"ffmpeg       : {ffmpeg_bin}")
    if args.retry_failed:
        print(f"Retry mode   : sources from {failed_log}")
    if args.recover:
        print("Decode       : tolerant only (--recover)")
    else:
        print("Decode       : strict, then tolerant retry on failure")
    if args.hwaccel_videotoolbox:
        print("HW decode    : VideoToolbox")
    print()

    if args.retry_failed:
        videos = _load_retry_sources(failed_log, input_roots)
        if not videos:
            print("No valid paths to retry (check failed log and --input_dirs).")
            sys.exit(1)
        print(f"Retrying {len(videos)} video(s) from failed log...\n")
    else:
        videos = _collect_videos(input_roots)
        if not videos:
            print("No videos found. Check --input_dirs.")
            sys.exit(1)
        print(f"Found {len(videos)} video(s). Starting re-encode...\n")

    # Map each video to its source root so we can reconstruct the output path
    def root_for(v: Path) -> Path:
        for r in sorted(input_roots, key=lambda r: len(r.parts), reverse=True):
            try:
                v.relative_to(r)
                return r
            except ValueError:
                continue
        return input_roots[0]

    tasks = [
        (v, _build_output_path(v, root_for(v), output_root))
        for v in videos
    ]

    ok_count = fail_count = skip_count = 0
    failed_sources: list[Path] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                reencode,
                ffmpeg_bin,
                src,
                dst,
                args.size,
                args.fps,
                args.crf,
                args.dry_run,
                skip_if_exists,
                args.recover,
                args.hwaccel_videotoolbox,
            ): src
            for src, dst in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            src, success, msg = future.result()
            status = "✓" if success else "✗"
            print(f"[{i:>4}/{len(tasks)}] {status} {src.name}  {msg}")
            if "skipped" in msg:
                skip_count += 1
            elif success:
                ok_count += 1
            else:
                fail_count += 1
                failed_sources.append(src)

    if not args.dry_run:
        if failed_sources:
            failed_log.parent.mkdir(parents=True, exist_ok=True)
            failed_log.write_text(
                "\n".join(str(p.resolve()) for p in sorted(set(failed_sources))) + "\n"
            )
            print(f"\nWrote {len(set(failed_sources))} failed path(s) to {failed_log}")
        elif failed_log.is_file():
            failed_log.unlink()
            print(f"\nRemoved failed log (no failures this run): {failed_log}")

    print(f"\nDone. {ok_count} re-encoded, {skip_count} skipped, {fail_count} failed.")
    if fail_count:
        print(
            "\nNote: Tolerant decode cannot fix a truncated MP4 or missing moov atom — "
            "re-export from QuickTime (File → Export) or replace the clip from source.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
