# Descreen

Mac-native drop target for descreening offset-printed CD front JPEGs (Pixelmator-processed: straightened, cropped, WhiBal’d). Writes a full-res `*-descreen.jpg` plus a 1000px-tall embed copy for FLAC tags.

Same layout as [ctdbfix](../ctdbfix): Python **engine** + AppKit **app**.

## Layout

- `engine/` — `descreen.py` CLI (`--json` for the app)
- `app/` — Mac app (drop / Open With a JPEG)
- `scripts/sync-engine.sh` — venv + copy into app Resources
- `scripts/build-app.sh` — build `app/build/Descreen.app`

## Quick start

Requires Homebrew Python 3 (`brew install python`) and Apple Command Line Tools (or Xcode).

```bash
./scripts/sync-engine.sh
./scripts/build-app.sh
open app/build/Descreen.app
```

Engine only:

```bash
cd engine && ./setup.sh
./descreen-engine /path/to/front.jpg
./descreen-engine --json /path/to/front.jpg
```

Do **not** run this on archival MusicBrainz cover JPEGs.

## License

Personal tooling. Algorithm lives in `engine/descreen.py`.
