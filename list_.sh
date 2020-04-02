#!/usr/bin/env bash

#####
    #
    # (c) Zarek Siegel
    # created 03/14/20 22:51

mktemp tree_out.fifo

python -c 'print(import )'


# tree -f covid19
    # | python \
        # -c \
            # 'print("hi")'


