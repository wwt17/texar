#!/bin/bash -x

prefix="nba."
suffix=".txt"

for stage in {train,valid,test}; do
	for ref_str in {"","_ref"}; do
		main_file=${prefix}sent${ref_str}.${stage}${suffix}
		raw_file=${prefix}sent${ref_str}.${stage}.raw${suffix}
		replaced_file=${prefix}sent${ref_str}.${stage}.replaced${suffix}
		mv $main_file $raw_file && \
		python3 replace_numbers.py < ${raw_file} > ${replaced_file} && \
		ln -s $replaced_file $main_file
	done
done

python3 make_vocabs.py
python3 join_data.py
