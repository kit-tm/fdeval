#!/bin/bash

# small helper script to execute a batch test

FIRSTPARAM=$1

tmux kill-session -t run_test

tmux new-session -d -s run_test

tmux send-keys -t run_test "echo "$FIRSTPARAM C-m
 
tmux send-keys -t run_test "source venv/bin/activate" C-m

tmux send-keys -t run_test "python test.py -n -t 24 -f " $FIRSTPARAM C-m