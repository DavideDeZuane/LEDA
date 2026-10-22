#!/bin/bash


valgrind --error-limit=no --suppressions=./build/bin/KEM-CPA/leda_ct.supp \
    ./build/bin/KEM-CPA/LEDAcrypt_KEM-CPA_SL_1_N0_2_ctcheck decoder

