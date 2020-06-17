#!/bin/bash

for value in {0..73}
do
	# echo "out"$value".ps"
	# ps2pdf out$value.ps ../PDF_OUTPUT/out$value.pdf
	pdftotext out$value.pdf ../TEXT_OUTPUT/out$value.txt
done