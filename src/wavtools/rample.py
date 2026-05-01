#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def convert_wav(input_path: Path, output_path: Path) -> None:
    cmd = [
        "sox",
        str(input_path),
        "-r", "44100",
        "-b", "16",
        "-c", "1",
        str(output_path),
    ]

    print(f"Converting: {input_path} -> {output_path}")
    subprocess.run(cmd, check=True)


def process_folder(folder: Path, overwrite: bool = False, delete_originals: bool = False) -> None:
    wav_files = sorted(folder.glob("*.wav"))

    if not wav_files:
        return

    if len(wav_files) != 4:
        print(f"Skipping {folder}: expected 4 .wav files, found {len(wav_files)}")
        return

    temp_dir = folder / ".converted_tmp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: convert all files to temp
        for i, wav_file in enumerate(wav_files, start=1):
            temp_output = temp_dir / f"{i}.wav"
            convert_wav(wav_file, temp_output)

        # Step 2: optionally delete originals
        if delete_originals:
            for wav_file in wav_files:
                wav_file.unlink()

        # Step 3: move converted files into place
        for i in range(1, 5):
            final_output = folder / f"{i}.wav"

            if final_output.exists():
                if overwrite:
                    final_output.unlink()
                else:
                    raise FileExistsError(
                        f"{final_output} already exists. Use --overwrite to replace it."
                    )

            (temp_dir / f"{i}.wav").replace(final_output)

    finally:
        # cleanup temp dir
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively convert folders of 4 .wav files to mono, 16-bit, 44100 Hz."
    )
    parser.add_argument("folder", help="Root folder")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 1.wav through 4.wav files",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Delete original .wav files after successful conversion",
    )

    args = parser.parse_args()
    root = Path(args.folder).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    for folder in [root, *root.rglob("*")]:
        if folder.is_dir():
            try:
                process_folder(
                    folder,
                    overwrite=args.overwrite,
                    delete_originals=args.delete_originals,
                )
            except Exception as e:
                print(f"Failed in {folder}: {e}")


if __name__ == "__main__":
    main()
