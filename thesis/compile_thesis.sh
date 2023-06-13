FILE=thesis.tex

# check if the directory is correct (has thesis.tex)
if [ -f "$FILE" ]; then
  # make sure output directory for pdf exists
  mkdir -p pdf_out

  # remove everything in the output directory except the thesis pdf
  find ./pdf_out/ -type f ! -name 'thesis.pdf' -delete

  # TeX loop
  pdflatex -output-directory=./pdf_out $FILE
  bibtex pdf_out/${FILE:0:-4}.aux
  pdflatex -output-directory=./pdf_out $FILE
  pdflatex -output-directory=./pdf_out $FILE

  # remove unnecessary artifacts
  find ./pdf_out/ -type f ! -name 'thesis.pdf' -delete
else
  echo "$FILE does not exist. Ensure the directory is correct."
fi
