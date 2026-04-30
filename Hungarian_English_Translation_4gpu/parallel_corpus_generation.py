from add_translation import add_translation
import argparse
import time
import gc
import torch

def main():
    """Run translation over dataset chunks using CLI arguments, then perform cleanup of memory and GPU resources."""
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    add_translation(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_existing=True,
    )

    print("Translation finished, cleaning up...")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    print("Clean exit.")

if __name__ == "__main__":
    main()