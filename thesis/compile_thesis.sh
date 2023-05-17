FILE=thesis.tex

if [ -f "$FILE" ]; then
  mkdir -p pdf_out
  pdflatex -output-directory=./pdf_out $FILE
  bibtex pdf_out/${FILE:0:-4}.aux
  pdflatex -output-directory=./pdf_out $FILE
  pdflatex -output-directory=./pdf_out $FILE
else
  echo "$FILE does not exist. Ensure the directory is correct."
fi
