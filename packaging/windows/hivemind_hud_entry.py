import multiprocessing
import sys

from hivemind.cli.standalone import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
