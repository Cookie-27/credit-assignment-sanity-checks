"""Capture real CLI output; optionally render it with the authoring-only Pillow dependency."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("A proxy can reverse the update direction", ["demo"],
     "The true gradient is positive; the smoothed proxy is negative in this case."),
    ("Equal-value drift need not reduce reward", ["demo", "--q0", "1", "--q1", "1"],
     "Both actions always succeed. Every displayed reward change is zero."),
    ("Complete action echo changes the result", ["demo", "--echo"],
     "Revealing the action makes the smoothed proxy positive in this case."),
    ("A ranking swap can preserve the total coefficient", ["coefficient-demo"],
     "This identity requires identical policy inputs and target tokens."),
)


def pillow_modules():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("GIF authoring requires Pillow; CLI experiments and text capture do not.")
    # Fonts are passed by the caller; the repository contains no machine paths.
    return Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="Also render docs/assets/demo.gif (requires Pillow).")
    parser.add_argument("--font", type=Path, help="Path to a local monospace TrueType font, required with --gif.")
    args = parser.parse_args()
    if args.gif and (args.font is None or not args.font.is_file()):
        parser.error("--gif requires --font pointing to a local monospace TrueType font")
    records = []
    for title, command, note in CASES:
        result = subprocess.run([sys.executable, "-B", "-m", "credit_sanity", *command],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
        if result.returncode:
            raise SystemExit(result.stderr)
        records.append({"title": title, "command": "python -m credit_sanity " + " ".join(command),
                        "output": result.stdout, "note": note})
    destination = ROOT / "docs/assets"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "demo-output.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    lines = ["# Captured CLI demonstration", "",
             "These are actual outputs captured by `python scripts/build_demo.py`.",
             "The accompanying GIF renders the same transcript, with five seconds per case;",
             "its playback timing does not represent program runtime.", "",
             "Run all commands from the repository root. See the [walkthrough](walkthrough.md)",
             "for Python examples and [research note](research_note.md) for assumptions.", ""]
    for index, item in enumerate(records, 1):
        lines += [f"## {index}. {item['title']}", "", "```bash", item["command"], "```", "",
                  "```text", item["output"].rstrip(), "```", "", item["note"], ""]
    lines += ["## Rebuild the presentation", "",
              "Text capture uses only the standard library. GIF authoring optionally needs",
              "Pillow and a local monospace TrueType font:", "", "```bash",
              "python scripts/build_demo.py --gif --font /path/to/monospace.ttf", "```", "",
              "Pillow is not a dependency of the experiment, its tests, or the walkthrough.", ""]
    (ROOT / "docs/demo.md").write_text("\n".join(lines), encoding="utf-8")
    if args.gif:
        Image, ImageDraw, ImageFont = pillow_modules()
        mono = ImageFont.truetype(str(args.font), 17)
        heading = ImageFont.truetype(str(args.font), 23)
        frames = []
        width, height = 1280, 790
        for index, item in enumerate(records, 1):
            frame = Image.new("RGB", (width, height), "#101827")
            draw = ImageDraw.Draw(frame)
            draw.text((36, 24), "CREDIT ASSIGNMENT SANITY CHECKS", fill="#80cbc4", font=heading)
            draw.text((36, 65), f"{index}/4  {item['title']}", fill="#f4f7fb", font=heading)
            draw.rounded_rectangle((24, 116, width - 24, 678), radius=12, fill="#1c2738")
            output_lines = ["$ " + item["command"], ""] + item["output"].splitlines()
            for row, line in enumerate(output_lines):
                y = 140 + row * 24
                box = draw.textbbox((44, y), line, font=mono)
                if box[2] > width - 44 or box[3] > 660:
                    raise ValueError("Captured output no longer fits the demo; adjust its layout")
                draw.text((44, y), line, fill="#83dbc4" if row == 0 else "#ecf0f5", font=mono)
            draw.text((36, 704), item["note"], fill="#f6cf80", font=mono)
            draw.text((36, 748), "Exact finite diagnostics | Captured CLI output | Rendered playback, 5 s per case",
                      fill="#a7b4c5", font=mono)
            frames.append(frame)
        frames[0].save(destination / "demo.png")
        frames[0].save(destination / "demo.gif", save_all=True, append_images=frames[1:],
                       duration=5000, loop=0, optimize=False)
    print(f"Captured {len(records)} successful commands; GIF rendered: {args.gif}.")


if __name__ == "__main__":
    main()
