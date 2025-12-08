#!/bin/bash
# Convert Academic Report Markdown to PDF with proper formatting
# Requires: pandoc, LaTeX distribution (TeX Live or MacTeX)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_MD="$PROJECT_ROOT/docs/ACADEMIC_REPORT.md"
OUTPUT_PDF="$PROJECT_ROOT/docs/ACADEMIC_REPORT.pdf"
TEMP_DIR="$PROJECT_ROOT/docs/temp_pdf_build"
TEMPLATE_FILE="$TEMP_DIR/template.tex"

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo -e "${RED}Error: pandoc is not installed${NC}"
    echo "Install with:"
    echo "  macOS: brew install pandoc"
    echo "  Ubuntu: sudo apt-get install pandoc"
    echo "  Or visit: https://pandoc.org/installing.html"
    exit 1
fi

# Check if LaTeX is installed (check for xelatex or pdflatex)
PDF_ENGINE=""
if command -v xelatex &> /dev/null; then
    PDF_ENGINE="xelatex"
    echo -e "${GREEN}Using XeLaTeX engine${NC}"
elif command -v pdflatex &> /dev/null; then
    PDF_ENGINE="pdflatex"
    echo -e "${GREEN}Using PDFLaTeX engine${NC}"
else
    echo -e "${RED}Error: No LaTeX engine found (xelatex or pdflatex)${NC}"
    echo "Install LaTeX distribution:"
    echo "  macOS: brew install --cask mactex"
    echo "  Ubuntu: sudo apt-get install texlive-xetex texlive-latex-extra"
    echo "  Or visit: https://www.latex-project.org/get/"
    exit 1
fi

# Create temp directory for build files
mkdir -p "$TEMP_DIR"

# Create a custom LaTeX template for better formatting
cat > "$TEMPLATE_FILE" << 'EOF'
\documentclass[11pt,a4paper]{article}

% Page geometry
\usepackage[top=1in,bottom=1in,left=1.5in,right=1in]{geometry}

% Math packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% For better font rendering
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}

% Images
\usepackage{graphicx}
\usepackage{float}

% Hyperlinks
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue,
    pdfauthor={Stephan Haloftis},
    pdftitle={GPU-Accelerated 2D Lattice Boltzmann Method for Aerodynamic Flow Simulation}
}

% Better code formatting
\usepackage{listings}
\usepackage{xcolor}
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

% Allow long lines in verbatim
\usepackage{fancyvrb}

% Header/Footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Better tables
\usepackage{booktabs}

% Title page
\title{GPU-Accelerated 2D Lattice Boltzmann Method\\for Aerodynamic Flow Simulation}
\author{Stephan Haloftis\\shaloft1@jhu.edu\\Johns Hopkins University}
\date{\today}

\begin{document}

\maketitle

$body$

\end{document}
EOF

echo -e "${YELLOW}Converting Markdown to PDF...${NC}"
echo "Input:  $REPORT_MD"
echo "Output: $OUTPUT_PDF"
echo ""

# Convert markdown to PDF using pandoc
pandoc "$REPORT_MD" \
    --from=markdown+tex_math_dollars+raw_tex+inline_code_attributes \
    --to=pdf \
    --pdf-engine="$PDF_ENGINE" \
    --template="$TEMPLATE_FILE" \
    --output="$OUTPUT_PDF" \
    --toc \
    --toc-depth=3 \
    --number-sections \
    --highlight-style=tango \
    --variable=geometry:margin=1in \
    --resource-path="$PROJECT_ROOT/docs:$PROJECT_ROOT/images:$PROJECT_ROOT" \
    --standalone \
    --metadata=date="$(date +'%B %Y')" \
    2>&1 | while IFS= read -r line; do
        # Filter out common warnings
        if [[ ! "$line" =~ "Package hyperref Warning" ]] && \
           [[ ! "$line" =~ "Overfull \\hbox" ]] && \
           [[ ! "$line" =~ "Underfull \\hbox" ]]; then
            echo "$line"
        fi
    done

# Check if PDF was created
if [ -f "$OUTPUT_PDF" ]; then
    PDF_SIZE=$(du -h "$OUTPUT_PDF" | cut -f1)
    echo ""
    echo -e "${GREEN}✓ PDF generated successfully!${NC}"
    echo -e "${GREEN}  Output: $OUTPUT_PDF${NC}"
    echo -e "${GREEN}  Size: $PDF_SIZE${NC}"
    echo ""
    
    # Clean up temp files
    rm -rf "$TEMP_DIR"
    
    exit 0
else
    echo -e "${RED}✗ Error: PDF was not generated${NC}"
    echo "Check the error messages above"
    rm -rf "$TEMP_DIR"
    exit 1
fi

