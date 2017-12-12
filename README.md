# README #

This repository is about data analysis of Single azurins while it undergoes electron transfer.

Most of the analysis can be found in the .ipynb files and many of the functions are in the .py files.

Two of the important notebooks can visited here:
[Manuscript_figures](https://github.com/biswajitSM/Azurin_SM_repo/blob/master/Manuscript/Figure/Manuscript_figures.ipynb), 
[Supporting_info](https://github.com/biswajitSM/Azurin_SM_repo/blob/master/Manuscript/Figure_SI/Supporting_info.ipynb)

***Manuscript Compilation***
To generate the pdf manuscript, go to the Manuscript folder and run the following codes in sequence:

	make svgtopdf #Run it once to convert svg files to pdf
	make build-all
The first command converts the svg files to pdf. This requires Python and Inkscape installed.

To clean the auxillary files:

	make clean
