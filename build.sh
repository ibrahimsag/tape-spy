#!/bin/sh
set -e
cc -O2 -o mnist_view main.c $(pkg-config --cflags --libs sdl3)
