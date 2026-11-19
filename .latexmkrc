# Keep build products beside each root document, inside its output directory.
# Changing to the document directory also makes relative images/includes work.
$do_cd = 1;
$pdf_mode = 1;
$out_dir = 'output';
$aux_dir = 'output';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
