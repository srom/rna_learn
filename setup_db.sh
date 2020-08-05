#!/bin/bash -e

DB_FOLDER="$(PWD)/data/condensed_traits/db"
DB_PATH="${DB_FOLDER}/seq.db"
if test -f "${DB_PATH}"; then
	echo "DB already exists, nothing to do"
	exit 0
fi

echo "=========================================="
echo "Creating database at ${DB_PATH}"
echo "=========================================="

mkdir -p ${DB_FOLDER}

python -m preprocessing.setup_db

echo "=========================================="
echo "Importing RNA sequences"
echo "=========================================="

python -m preprocessing.import_rna_into_db

echo "=========================================="
echo "Importing coding sequences"
echo "=========================================="

python -m preprocessing.import_cds_into_db

echo "=========================================="
echo "Importing non-coding sequences"
echo "=========================================="

python -m preprocessing.import_non_coding_into_db

echo "=========================================="
echo "Database ready"
echo "=========================================="

exit 0
