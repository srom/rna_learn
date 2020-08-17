#!/bin/bash -e

PEMKEY=$1
IP=$2
RUNID=$3

echo "Syncing remote logs for run ${RUNID} from ${IP}"
rsync -avzhe "ssh -i ${PEMKEY}" ubuntu@$IP:/home/ubuntu/rna_learn/summary_log/$RUNID summary_log/

echo "Launching tensorboard for run ${RUNID}"
tensorboard --logdir=summary_log/$RUNID
