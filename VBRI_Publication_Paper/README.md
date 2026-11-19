# Building the VBRI paper

Open `main.tex` in VS Code and run **LaTeX Workshop: Build LaTeX project**.
The default workspace recipe runs latexmk, which runs pdfLaTeX and Biber
and repeats the required steps to resolve citations and references.

The PDF is generated at `output/main.pdf`. Auxiliary files, logs, and
SyncTeX files also go into `output/`, which is created automatically.
LaTeX Workshop's PDF viewer uses that same output directory.

From this directory, the equivalent terminal command is:

```bash
latexmk main.tex
```

From the parent BPD directory:

```bash
latexmk VBRI_Publication_Paper/main.tex
```

Both directories contain a `.latexmkrc` that selects PDF output, changes
to the root document's directory, and stores build files in `output/`.
The BPD workspace settings apply the same layout to other LaTeX documents
in this workspace. The VS Code user settings also use `output/` by default.

For TinyTeX installations, the packages additionally required by this paper
can be installed with:

```bash
tlmgr install preprint makecell microtype pgf biblatex biber enumitem caption booktabs multirow
```

`preprint` supplies `authblk.sty`; `pgf` supplies TikZ. The paper uses Biber
through biblatex. Use the default latexmk recipe to build its bibliography.

To clear generated intermediate files while keeping the PDF:

```bash
latexmk -c main.tex
```
