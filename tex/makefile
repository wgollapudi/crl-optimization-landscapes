MAIN := proposal
BUILDDIR := build

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -bibtex -interaction=nonstopmode -file-line-error

TARGETS := proposal proposal_updated

.PHONY: all watch clean distclean $(TARGETS)

all: $(TARGETS)

$(TARGETS):
	mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -outdir=$(BUILDDIR) $@.tex

watch:
	mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc -outdir=$(BUILDDIR) proposal.tex

clean:
	$(LATEXMK) -c -outdir=$(BUILDDIR) proposal.tex
	$(LATEXMK) -c -outdir=$(BUILDDIR) proposal_updated.tex

distclean:
	$(LATEXMK) -C -outdir=$(BUILDDIR) proposal.tex
	$(LATEXMK) -C -outdir=$(BUILDDIR) proposal_updated.tex
	rm -rf $(BUILDDIR)
