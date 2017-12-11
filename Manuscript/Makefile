all:
	pdflatex -interaction=nonstopmode Sup_info.tex
	pdflatex -interaction=nonstopmode azurin.tex
	evince azurin.pdf &
	evince Sup_info.pdf &
build-all:
	pdflatex -interaction=nonstopmode Sup_info.tex
	# bibtex Sup_info
	pdflatex -interaction=nonstopmode transbinding.tex
	bibtex azurin
	pdflatex -interaction=nonstopmode Sup_info.tex
	pdflatex -interaction=nonstopmode Sup_info.tex
	pdflatex -interaction=nonstopmode azurin.tex
	pdflatex -interaction=nonstopmode azurin.tex
	evince azurin.pdf &
	evince Sup_info.pdf &	
read:
	evince azurin.pdf &
	evince Sup_info.pdf &
svgtopdf:
	python all_svg2pdf.py
clean-all:
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~
clean:
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~