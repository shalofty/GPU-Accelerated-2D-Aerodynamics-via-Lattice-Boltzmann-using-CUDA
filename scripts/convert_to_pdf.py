#!/usr/bin/env python3
"""
Convert Academic Report Markdown to PDF with proper formatting
Uses pandoc for conversion with LaTeX math rendering

Requirements:
    - pandoc: https://pandoc.org/installing.html
    - LaTeX distribution (TeX Live, MacTeX, or MiKTeX)
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
REPORT_MD = PROJECT_ROOT / "docs" / "ACADEMIC_REPORT.md"
OUTPUT_PDF = PROJECT_ROOT / "docs" / "ACADEMIC_REPORT.pdf"
TEMP_DIR = PROJECT_ROOT / "docs" / "temp_pdf_build"
TEMPLATE_FILE = TEMP_DIR / "template.tex"


def check_command(cmd):
    """Check if a command is available."""
    return shutil.which(cmd) is not None


def create_latex_template():
    """Create a custom LaTeX template for better formatting."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    template_content = r'''\documentclass[11pt,a4paper]{article}

% Page geometry
\usepackage[top=1in,bottom=1in,left=1.5in,right=1in]{geometry}

% Math packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% For better font rendering (XeLaTeX only)
\ifxetex
    \usepackage{fontspec}
    % Use system fonts (available on macOS)
    \setmainfont{Times New Roman}
    \setsansfont{Helvetica}
    \setmonofont{Courier New}
\fi

% Images
\usepackage{graphicx}
\usepackage{float}
\graphicspath{{./images/}{./docs/images/}{./}}

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
    backgroundcolor=\color{gray!10},
    language=C++,
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{green!60!black},
    stringstyle=\color{red},
    showstringspaces=false
}

% Allow long lines in verbatim
\usepackage{fancyvrb}

% Shaded and Highlighting environments for code blocks (required by pandoc)
\usepackage{framed}
\usepackage{xcolor}
\definecolor{Shaded}{RGB}{248,248,248}
\let\Shaded\framed
\let\endShaded\endframed

% Define Highlighting environment (used by pandoc for syntax highlighting)
\newenvironment{Highlighting}{}{}
\newcommand{\KeywordTok}[1]{\textcolor[rgb]{0.00,0.44,0.13}{\textbf{{#1}}}}
\newcommand{\DataTypeTok}[1]{\textcolor[rgb]{0.56,0.13,0.00}{{#1}}}
\newcommand{\DecValTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\BaseNTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\FloatTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\CharTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\StringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\CommentTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textit{{#1}}}}
\newcommand{\OtherTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{{#1}}}
\newcommand{\AlertTok}[1]{\textcolor[rgb]{0.94,0.16,0.16}{{#1}}}
\newcommand{\FunctionTok}[1]{\textcolor[rgb]{0.00,0.00,1.00}{{#1}}}
\newcommand{\RegionMarkerTok}[1]{{#1}}
\newcommand{\ErrorTok}[1]{\textcolor[rgb]{0.64,0.00,0.00}{\textbf{{#1}}}}
\newcommand{\NormalTok}[1]{{#1}}
\newcommand{\OperatorTok}[1]{\textcolor[rgb]{0.81,0.36,0.00}{\textbf{{#1}}}}
\newcommand{\ControlFlowTok}[1]{\textcolor[rgb]{0.00,0.44,0.13}{\textbf{{#1}}}}
\newcommand{\ImportTok}[1]{{#1}}
\newcommand{\DocumentationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{{#1}}}}
\newcommand{\InformationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{{#1}}}}
\newcommand{\WarningTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{{#1}}}}
\newcommand{\VariableTok}[1]{\textcolor[rgb]{0.00,0.27,0.87}{{#1}}}
\newcommand{\AttributeTok}[1]{\textcolor[rgb]{0.49,0.56,0.16}{{#1}}}
\newcommand{\BuiltInTok}[1]{{#1}}
\newcommand{\ConstantTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\PreprocessorTok}[1]{\textcolor[rgb]{0.74,0.48,0.00}{{#1}}}
\newcommand{\ExtensionTok}[1]{{#1}}
\newcommand{\SpecialCharTok}[1]{\textcolor[rgb]{0.25,0.44,0.63}{{#1}}}
\newcommand{\VerbatimStringTok}[1]{\textcolor[rgb]{0.25,0.44,0.63}{{#1}}}

% Header/Footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Better tables
\usepackage{booktabs}

% Define tightlist for pandoc (required for lists)
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

% Define pandoc-specific commands that might be missing
\providecommand{\pandocbounded}[1]{#1}

% Title page
\title{GPU-Accelerated 2D Lattice Boltzmann Method\\for Aerodynamic Flow Simulation}
\author{Stephan Haloftis\\shaloft1@jhu.edu\\Johns Hopkins University}
\date{\today}

\begin{document}

\maketitle

$body$

\end{document}
'''
    
    TEMPLATE_FILE.write_text(template_content)
    print(f"Created LaTeX template: {TEMPLATE_FILE}")


def find_pdf_engine():
    """Find available PDF engine."""
    engines = ["xelatex", "pdflatex", "lualatex"]
    for engine in engines:
        if check_command(engine):
            return engine
    return None


def preprocess_math(md_content):
    """Preprocess markdown to convert backslash-parentheses math to dollar signs for pandoc."""
    import re
    # Convert \(...\) to $...$ (inline math)
    # Need to escape properly: \( becomes \\( in regex
    md_content = re.sub(r'\\\(', '$', md_content)
    md_content = re.sub(r'\\\)', '$', md_content)
    # Convert \[...\] to $$...$$ (display math)
    md_content = re.sub(r'\\\[', '$$', md_content)
    md_content = re.sub(r'\\\]', '$$', md_content)
    return md_content


def convert_to_pdf():
    """Convert markdown to PDF using pandoc."""
    
    # Check prerequisites
    if not check_command("pandoc"):
        print("ERROR: pandoc is not installed")
        print("Install with:")
        print("  macOS: brew install pandoc")
        print("  Ubuntu: sudo apt-get install pandoc")
        print("  Or visit: https://pandoc.org/installing.html")
        sys.exit(1)
    
    pdf_engine = find_pdf_engine()
    if not pdf_engine:
        print("ERROR: No LaTeX engine found (xelatex, pdflatex, or lualatex)")
        print("Install LaTeX distribution:")
        print("  macOS: brew install --cask mactex")
        print("  Ubuntu: sudo apt-get install texlive-xetex texlive-latex-extra")
        print("  Or visit: https://www.latex-project.org/get/")
        sys.exit(1)
    
    print(f"Using PDF engine: {pdf_engine}")
    
    # Create template
    create_latex_template()
    
    # Check if input file exists
    if not REPORT_MD.exists():
        print(f"ERROR: Input file not found: {REPORT_MD}")
        sys.exit(1)
    
    # Preprocess markdown to normalize math syntax
    print("Preprocessing math syntax...")
    md_content = REPORT_MD.read_text(encoding='utf-8')
    processed_content = preprocess_math(md_content)
    
    # Write processed content to temp file
    temp_md = TEMP_DIR / "processed_report.md"
    temp_md.write_text(processed_content, encoding='utf-8')
    
    print(f"\nConverting Markdown to PDF...")
    print(f"Input:  {REPORT_MD}")
    print(f"Output: {OUTPUT_PDF}")
    print()
    
    # Build pandoc command
    # Important: Add both docs and docs/images for relative paths
    resource_paths = [
        str(PROJECT_ROOT / "docs"),
        str(PROJECT_ROOT / "docs" / "images"),
        str(PROJECT_ROOT / "images"),
        str(PROJECT_ROOT),
    ]
    
    # Lua filter for math handling (optional)
    filter_file = SCRIPT_DIR / "fix_math_for_pdf.lua"
    
    cmd = [
        "pandoc",
        str(temp_md),  # Use preprocessed file
        "--from=markdown+tex_math_dollars+latex_macros+raw_tex+inline_code_attributes",
        "--to=pdf",
        f"--pdf-engine={pdf_engine}",
        f"--template={TEMPLATE_FILE}",
        f"--output={OUTPUT_PDF}",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
        "--highlight-style=tango",
        "--variable=geometry:margin=1in",
        "--resource-path=" + ":".join(resource_paths),
        "--standalone",
        f"--metadata=date={Path.cwd()}",
    ]
    
    # Add filter if it exists (optional enhancement)
    if filter_file.exists():
        cmd.extend(["--lua-filter", str(filter_file)])
    
    # Run pandoc
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        # Filter out common warnings
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line and not any(x in line for x in [
                    "Package hyperref Warning",
                    "Overfull \\hbox",
                    "Underfull \\hbox",
                    "Package caption Warning",
                ]):
                    print(line, file=sys.stderr)
        
        if result.returncode != 0:
            print("ERROR: pandoc conversion failed")
            if result.stdout:
                print(result.stdout)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to run pandoc: {e}")
        sys.exit(1)
    
    # Check if PDF was created
    if OUTPUT_PDF.exists():
        pdf_size = OUTPUT_PDF.stat().st_size / (1024 * 1024)  # MB
        print(f"\n✓ PDF generated successfully!")
        print(f"  Output: {OUTPUT_PDF}")
        print(f"  Size: {pdf_size:.2f} MB")
        
        # Clean up temp files
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        
        return 0
    else:
        print("ERROR: PDF was not generated")
        print("Check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(convert_to_pdf())

