#!/bin/bash -e

PEMKEY=$1
IP=$2
RUNID=$3

rsync -avzhe "ssh -i ${PEMKEY}" ubuntu@$IP:/home/ubuntu/rna_learn/summary_log/$RUNID summary_log/
