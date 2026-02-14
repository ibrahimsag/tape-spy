#!/bin/sh
set -e
cc -O2 -o main main.c $(pkg-config --cflags --libs sdl3)
