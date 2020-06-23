#!/bin/bash

# small helper script to execute a batch read

FIRSTPARAM=$1

tmux kill-session -t run_test

tmux new-session -d -s run_test

tmux send-keys -t run_test "echo "$FIRSTPARAM C-m
 
tmux send-keys -t run_test "source venv/bin/activate" C-m

tmux send-keys -t run_test "python main.py -t 24 -r " $FIRSTPARAM C-m