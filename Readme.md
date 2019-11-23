#des-crack

This program performs DES cracking using brute force and CUDA.

Requirements
---
Computer with Nvidia GPU, Linux, CUDA toolkit (tested version 10.1), GCC

Building
---
Check your CUDA installation and (If needed) adjust CUDAPATH variable in Makefile (first line). Then just use make.

Running
---
To check possible options run program with flag "-h".

Measuring performance:
---
To compare performace between CPU and GPU you can first run:

    ./des-crack -s 26 -c

to crack 26-bit key using one CPU thread (~90s on my Xeon e3-1230v2). Then:

    ./des-crack -s 26

to do this on GPU (~0.75s on my GTX 1060)
