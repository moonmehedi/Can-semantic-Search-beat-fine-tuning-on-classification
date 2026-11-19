# Apply the same output layout when building directly from this directory.
$do_cd = 1;
$pdf_mode = 1;
$out_dir = 'output';
$aux_dir = 'output';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
