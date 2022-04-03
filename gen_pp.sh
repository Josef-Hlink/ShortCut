#!/bin/bash

mkdir all
cd src

for i in `seq 50`; do
	python3 ShortCutExperiment.py
	
	cd ../results
	for file in `ls`; do
		mv ${file} ../all/${i}${file}
	done
	cd ../src
done
