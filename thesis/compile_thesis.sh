FILE=thesis.tex
if [ -f "$FILE" ]; then
  mkdir -p pdf_out
  pdflatex -output-directory=./pdf_out thesis.tex
  bibtex pdf_out/thesis.aux
  pdflatex -output-directory=./pdf_out thesis.tex
  pdflatex -output-directory=./pdf_out thesis.tex
else
  echo "$FILE does not exist. Ensure the directory is correct."
fi
