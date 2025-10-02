% citask.cls
\ProvidesClass{citask}[2023/06/14 v1.0 CI TASK Publishing Class]

% -- Options --

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}

% Define custom style
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{paratype}
\RequirePackage{caption}
\RequirePackage{graphicx} % For including images
\RequirePackage{subcaption} % For subfigures
\RequirePackage{amsmath} % For math equations
\RequirePackage{amssymb} % For math symbols
\RequirePackage{amsthm} % For theorems
\RequirePackage{hyperref} % For hyperlinks
\RequirePackage[superscript,biblabel]{cite}
\RequirePackage{booktabs} % For tables
\RequirePackage{placeins}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{wrapfig}
\RequirePackage{float}
\RequirePackage[sort, numbers]{natbib}
\RequirePackage{listofitems}
\RequirePackage{ragged2e}
\RequirePackage{background}
\RequirePackage{fancyhdr}
\RequirePackage[percent]{overpic}
\RequirePackage{enumitem}
\RequirePackage{mathptmx}
\RequirePackage{prettyref}
\RequirePackage[T1]{fontenc}
\RequirePackage{PTSansNarrow}
\RequirePackage{PTSans}
\RequirePackage{libertine}
\RequirePackage{xfakebold}


% -- Settings --

\geometry{%
    top=0.5in,
    bottom=1in,
    left=0.5in,
    right=0.5in,
    a4paper
}
\setlength{\columnsep}{5mm}

% -- Styling --

\setlength{\emergencystretch}{3em}

\newcommand{\fbseries}{\unskip\setBold\aftergroup\unsetBold\aftergroup\ignorespaces}
\newcommand{\setBoldness}[1]{\def\fake@bold{#1}}

\definecolor{taskcolorprimary}{HTML}{003767}
\definecolor{taskcolorsecondary}{HTML}{DF004C}
\DeclareCaptionFont{taskcolorprimary}{\color{taskcolorprimary}}
\DeclareCaptionFont{taskcolorsecondary}{\color{taskcolorsecondary}}

\captionsetup[figure]{font=footnotesize, labelfont={small,taskcolorsecondary,bf}}
\captionsetup[table]{font=footnotesize, labelfont={small,taskcolorsecondary,bf}}
\captionsetup[subfigure]{font={bf,footnotesize}, skip=1pt, singlelinecheck=false}

\renewcommand\ttdefault{cmvtt}

\newrefformat{fig}{\textbf{\color{taskcolorprimary}Fig.~\ref{#1}}}
\newrefformat{tab}{\textbf{\color{taskcolorprimary}Tab.~\ref{#1}}}
\newrefformat{eq}{\textbf{\color{taskcolorprimary}Eq.~\ref{#1}}}

% Fonts

% Make the text fit the whole line width

\setlength{\parindent}{2em}
\setlength{\parskip}{0.1em}

% Set the hyperlink colors

\hypersetup{%
    colorlinks=true,
    linkcolor=taskcolorprimary,
    filecolor=black,
    urlcolor=black,
    citecolor=taskcolorsecondary,
    pdfborder={0 0 0}
}

% Format section titles

\renewcommand{\section}{\@startsection{section}{1}{\z@}%
             {0.05in}%
             {0.15in}%
             {\singlespacing\color{taskcolorprimary}\usefont{T1}{PTSansNarrow-TLF}{b}{n}\huge}}

\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
             {0.05in}%
             {0.10in}%
             {\singlespacing\color{taskcolorprimary}\usefont{T1}{PTSansNarrow-TLF}{b}{n}\Large}}

\renewcommand{\subsubsection}{\@startsection{subsubsection}{2}{\z@}%
             {0.05in}%
             {0.05in}%
             {\singlespacing\color{taskcolorprimary}\usefont{T1}{PTSansNarrow-TLF}{b}{n}\large}}

\def\@seccntformat#1{\@ifundefined{#1@cntformat}%
   {\csname the#1\endcsname\space}%    default
   {\csname #1@cntformat\endcsname}}%  enable individual control
\newcommand\section@cntformat{\thesection.\space}       % section-level
\newcommand\subsection@cntformat{\thesubsection.\space} % subsection-level

% Set text inside table cells to be always of footnotesize

\let\oldtabular\tabular
\renewcommand{\tabular}{\footnotesize\oldtabular}

% Set the table caption to be always on top

\floatstyle{plaintop}
\restylefloat{table}

\renewcommand\floatpagefraction{.9}
\renewcommand\dblfloatpagefraction{.9} % for two column documents
\renewcommand\topfraction{.9}
\renewcommand\dbltopfraction{.9} % for two column documents
\renewcommand\bottomfraction{.9}
\renewcommand\textfraction{.1}
\setcounter{totalnumber}{50}
\setcounter{topnumber}{50}
\setcounter{bottomnumber}{50}

% Front page

\newcommand{\keywords}[1]{%
  \gdef\@keywords{%
    {%
      \color{taskcolorprimary}
      \usefont{T1}{PTSansNarrow-TLF}{b}{n}
      \flushleft
      \small\bfseries\Large Keywords:
      \vspace{0.25em}
    }\\
    #1
  }
}

\newcommand{\MONTH}{%
  \ifcase\the\month
  \or January% 1
  \or February% 2
  \or March% 3
  \or April% 4
  \or May% 5
  \or June% 6
  \or July% 7
  \or August% 8
  \or September% 9
  \or October% 10
  \or November% 11
  \or December% 12
  \fi}
\def\mydate{\leavevmode\hbox{\twodigits\day~\MONTH~\the\year}}
\def\twodigits#1{\ifnum#1<10 0\fi\the#1}

\renewcommand{\author}[3]{%
  {% Author name
    \color{taskcolorprimary}
    \bfseries
    \flushleft
    \normalsize{#1}
    \vspace{0.1em}
  }\\
  {% Affiliation unit name
    \footnotesize
    \readlist*\affiliationname{#2}
    \foreachitem\x\in\affiliationname[]{%
      \ifnum\xcnt=1\else\,\\ \fi\x
    }
    \vspace{0.1em}
  }\\
  {% Affiliation address
    \footnotesize
    #3
  }\\
  \vspace{-1.75em}
}

\renewcommand{\abstract}[1]{%
  \gdef\@abstract{%
    {%
      \begin{flushleft}
        \bfseries\color{taskcolorprimary}
        \Large
        \usefont{T1}{PTSansNarrow-TLF}{b}{n}
        \abstractname
        \vspace{-1.5em}
      \end{flushleft}
      \begin{justify}
        \onehalfspacing
        \selectfont
        #1
      \end{justify}
    }
  }
}

\newcommand{\authors}[1]{
  \gdef\@authors{#1}
}

\usepackage[pscoord]{eso-pic}% The zero point of the coordinate systemis the lower left corner of the page (the default).

\newcommand{\placetextbox}[3]{% \placetextbox{<horizontal pos>}{<vertical pos>}{<stuff>}
  \setbox0=\hbox{#3}% Put <stuff> in a box
  \AddToShipoutPictureFG*{% Add <stuff> to current page foreground
    \put(\LenToUnit{#1\paperwidth},\LenToUnit{#2\paperheight}){\vtop{{\null}\makebox[0pt][c]{#3}}}%
  }%
}%

\newcommand{\makefrontpage}{%
  \backgroundsetup{%
    scale=1,
    angle=0,
    opacity=1.0,
    contents = {}
  }
  \vspace*{-0.75in}
  \centerline{%
    \includegraphics[width=1.01\paperwidth]{assets/titlepage_header}
  }
  \placetextbox{0.84}{0.845}{{
    \usefont{T1}{PTSansNarrow-TLF}{b}{n}
    \color{white}
    \setBoldness{0.5}
    \fbseries
    \large
    \MIR{TASK Quarterly 28 (3) 2025}
  }}%
  \begin{flushright}
  \end{flushright}
  \vspace{-0.75in}
  \thispagestyle{empty}
  {
      \usefont{T1}{PTSansNarrow-TLF}{m}{c}
      \begin{flushleft}
        \color{taskcolorprimary}
        \singlespacing
        \huge
        \bfseries
        \vspace{1em}
        \@title
        \vspace{-1.5em}
      \end{flushleft}
      \vspace{2em}
    {
      \usefont{T1}{PTSans-TLF}{m}{n}
      \@authors
    }
  }
  \vfill
  \@abstract
  \vfill
  \@keywords
  \vspace{1em}
  \pagebreak
  \twocolumn
}

% Customize footer using fancyhdr

\pagestyle{fancy}

\fancyhf{} % sets both header and footer to nothing
\renewcommand{\headrulewidth}{0pt}
\fancyhead{}

\fancyfoot{}
\lfoot[LE,LO]{%
  \begin{overpic}[width=1.035\textwidth]{assets/footer}
  \put (95,0) {\color{taskcolorprimary}$\mathbf{\thepage}$}
  \end{overpic}
}

\fancypagestyle{plain}{%
  \pagestyle{fancy}
}

% Set the default theme for booktabs tables

\setlength{\heavyrulewidth}{0.1em}
\setlength{\lightrulewidth}{0.05em}
\setlength{\cmidrulewidth}{0.05em}
\setlength{\belowrulesep}{0.25em}
\setlength{\belowbottomsep}{0.1em}
\setlength{\aboverulesep}{0.25em}
\setlength{\abovetopsep}{0.1em}
\setlength{\cmidrulesep}{0.1em}
\setlength{\cmidrulekern}{0.1em}
\setlength{\defaultaddspace}{0.1em}

\setlist{nolistsep}

% Set default list style

\setlist[itemize]{%
  label={\color{taskcolorsecondary}\tiny$\blacktriangleright$},
  leftmargin=1.5em,
  topsep=0.5em,
  partopsep=0em,
  parsep=0em,
  itemsep=0em,
  labelsep=0.5em
}

% Force figures in subsubsections

\AtBeginDocument{%
  \expandafter\renewcommand\expandafter\subsubsection\expandafter
    {\expandafter\@fb@secFB\subsubsection}%
  \newcommand\@fb@secFB{\FloatBarrier
    \gdef\@fb@afterHHook{\@fb@topbarrier \gdef\@fb@afterHHook{}}}%
  \g@addto@macro\@afterheading{\@fb@afterHHook}%
  \gdef\@fb@afterHHook{}%
}

\singlespacing

% Define references command

\makeatletter
\renewcommand*{\@biblabel}[1]{[#1]\hfill}
\makeatother
\setlength{\bibsep}{2pt plus 0.1em}

\newcommand{\references}[1]{%
  \footnotesize
  \bibliographystyle{ieeetr}
  \bibliography{#1}
  \normalsize
  \singlespacing
}
