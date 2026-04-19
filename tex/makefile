MAIN := proposal
BUILDDIR := build

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -bibtex -interaction=nonstopmode -file-line-error

.PHONY: all pdf watch clean distclean

all: pdf

pdf:
	mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -outdir=$(BUILDDIR) $(MAIN).tex

watch:
	mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -outdir=$(BUILDDIR) $(MAIN).tex

clean:
	$(LATEXMK) -c -outdir=$(BUILDDIR) $(MAIN).tex

distclean:
	$(LATEXMK) -C -outdir=$(BUILDDIR) $(MAIN).tex
	rm -rf $(BUILDDIR)
