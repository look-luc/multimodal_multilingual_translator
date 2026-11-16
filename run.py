import argparse
import gc

import torch

from src.main import main

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Run different elements of pipeline")
    parser.add_argument('--train', action='store_true', help='Training pipeline')

    args = parser.parse_args()
    main(args.train)
