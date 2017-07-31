README.txt 
==========

RIPS Document Template, needed to create the final report in LaTeX
===========================================================================
This file and all the ones referred to can be found in the "LaTeX Template"
subfolder of the "Templates-etc" folder on IPAM's R-Drive for the RIPS 
program.

It is included with several LaTeX files and subordinate folders that your 
team's Report Coordinator (RC) can copy to your team's report folder to 
begin the process of constructing your team's Final Report. 

The main file is "z-Report-Master-2015.tex", referred to here as the Master 
TeX file. This file, and a pdf of the typeset document, are prepended with
"z-" so that you can easily find them in the listing. The pdf explains in
detail the process of report preparation, as well as how to use the tex 
file.  In that respect, the tex file is self-referential, serving as both a 
template for a report in the "RIPS House Style" and a guide for writing the
report

You will find here several .tex files, to create chapters, appendices, 
glossary, with names like "chapter1.tex", "a-glossary.tx", along
with numerous ancillary files and a few folders. The Graphics folder 
contains all the graphics files.  Not included here are the intermediate 
files the typesetter creates while constructing the final product, your 
document --- files with extensions such as ".aux" and ".toc"; they will 
reappear when the typesetter is run again.

The template structure may seem intricate at first, but it has been designed
to help your team streamline the incremental process of constructing the 
Final Report.  Your team will need careful stewardship of the process, and 
that is the job of the RC.

This template was originally developed using PCTeX and is now being 
maintained using the TeXworks "pdfLaTeX" typesetter, available as an IPAM
option in the system's All Programs menu, where you can choose your own
default. The template can be used with other LaTeX typsetting programs, with
some qualifications.  For example, pdfLaTex requires graphics coded as pdf
files, which can be generated automatically from .eps files using 
appropriate code in the "preamble" of the Master Tex file.  

(A sample LaTeX presentation is also available in a folder nearby, called 
"beamer_presentation.tex", which may be of use to you if your team plans to 
write your midterm and final presentations in LaTeX/Beamer.)

To open and edit the LaTeX report, you can invoke the default typesetter by
clicking on your copy of the Master Tex file. Whichever typesetter you have 
chosen, be sure to read the "preamble" to make sure you've incorporated the 
required "package(s)" for that typesetter.

Academic Mentors (AMs) will present a two-part tutorial during the second week
of the RIPS program to introduce the LaTeX report template.  The afternoon 
session is devoted to hands-on practice for the RCs.  The AMs will also serve 
as LaTeX consultants.  Abundant help is available on the internet and in
specialized LaTeX manuals.

Please let me know of any problems you encounter with documentation.

-Mike Raugh

June 19, 2015
