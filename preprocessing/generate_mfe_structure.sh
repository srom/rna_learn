#!/bin/bash

PATTERN=$1

total=0
for path_in in ${PATTERN}; do
	total=$((total+1))
done

counter=0
for path_in in ${PATTERN}; do
	counter=$((counter+1))
	echo "${path_in} (${counter} / ${total})"
	path_out=$(echo ${path_in} | sed s/\.fasta/\.structure\.txt/)
    cat ${path_in} | RNAfold --noPS | tail -1 > ${path_out}
done
