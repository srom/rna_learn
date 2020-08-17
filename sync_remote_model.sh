#!/bin/bash -e

PEMKEY=$1
IP=$2
RUNID=$3

echo "Syncing model for run ${RUNID} from ${IP}"
rsync -avzhe "ssh -i ${PEMKEY}" ubuntu@$IP:/home/ubuntu/rna_learn/saved_models/$RUNID saved_models/
