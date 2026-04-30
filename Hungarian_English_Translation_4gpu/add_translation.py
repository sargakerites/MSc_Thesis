import os
import re
import json
from tqdm import tqdm

from translation import translation_batch


_CHUNK_RE = re.compile(r".*?(\d+)\.json$") 


def _extract_chunk_id(filename: str, fallback: int) -> str:
    """Extract a numeric chunk id from a filename, or return the fallback index if no id is found."""

    m = _CHUNK_RE.match(filename)
    if m:
        return m.group(1)
    return str(fallback)


def add_translation(
    input_folder: str,
    output_folder: str,
    *,
    batch_size: int = 32,
    start_idx: int = 0,
    end_idx: int | None = None,
    skip_existing: bool = True,
):
    """Translate missing sentence entries in JSON chunks and save the completed parallel corpus files."""
    
    os.makedirs(output_folder, exist_ok=True)

    tqdm.monitor_interval = 0

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])

    if end_idx is None:
        end_idx = len(files)

    start_idx = max(0, start_idx)
    end_idx = min(len(files), end_idx)

    print(f"Total files: {len(files)} | Processing slice: [{start_idx}, {end_idx})")

    for file_i in range(start_idx, end_idx):
        filename = files[file_i]

        input_path = os.path.join(input_folder, filename)

        chunk_id = _extract_chunk_id(filename, fallback=file_i)
        output_filename = f"parallel_corpus_{chunk_id}.json"
        output_path = os.path.join(output_folder, output_filename)
        tmp_path = output_path + ".tmp"

        if skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"SKIP: {output_filename} exists")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            data_chunk = json.load(f)

        total = len(data_chunk)

        with tqdm(
            total=total,
            desc=f"Translating {filename}",
            unit="sent",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.05,
        ) as pbar:

            for start in range(0, total, batch_size):
                batch = data_chunk[start:start + batch_size]

                idx_to_translate = []
                dicts_to_translate = []

                for local_i, d in enumerate(batch):
                    existing = d.get("translation", "")
                    if (not isinstance(existing, str)) or (not existing.strip()):
                        idx_to_translate.append(local_i)
                        dicts_to_translate.append(d)

                if dicts_to_translate:
                    outs = translation_batch(
                        dicts_to_translate,
                        batch_size=len(dicts_to_translate),
                    )
                    for local_i, out in zip(idx_to_translate, outs):
                        batch[local_i]["translation"] = out

                processed = len(batch)
                pbar.update(processed)

                done = pbar.n
                left = total - done
                elapsed = pbar.format_dict["elapsed"]
                avg = (elapsed / done) if done else 0.0
                remaining = avg * left

                rem_h = int(remaining // 3600)
                rem_m = int((remaining % 3600) // 60)
                rem_s = int(remaining % 60)
                remaining_s = f"{rem_h:02d}:{rem_m:02d}:{rem_s:02d}"

                pbar.set_postfix_str(
                    f"done={done} left={left} eta={remaining_s} avg={avg:.2f}s/sent",
                    refresh=False
                )

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data_chunk, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)

        print(f"WROTE: {output_filename}")