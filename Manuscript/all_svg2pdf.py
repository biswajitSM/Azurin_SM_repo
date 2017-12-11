#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
==============================================================================
                                all_svg2pdf

    A Python module to convert all .svg files in a root folder (and below) 
    to .pdf according to the following inkscape options:
    --without-gui
    --export-area-drawing

    For more inkscape options see: https://inkscape.org/sk/doc/inkscape-man.html

    Usage: 
    $python all_svg2pdf.py
==============================================================================
'''
import os

def getPaths():
    """Get root directory and define extension"""

    cwd = os.getcwd()
    extension = '.svg'

    return cwd, extension

def getSVGList(root, extension):
    """Get all svg files in root directory and below"""

    svg_list = []

    for path, subdirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                abspath_file = path + os.sep + file
                svg_list.append(abspath_file)
    print('Were found %i svg files under root path: %s' %(len(svg_list), root))
    return svg_list
                
def svg2pdf(list):
    """Convert all svg files to pdf according to specified options"""

    options = '-d 300 --without-gui --export-area-page'
    print('\nConverting, please wait...\n')
    for file in list:
        print('Converting file: %s' %file)
        pdf_abspath = file.split('.')[0]
        os.system('inkscape %s "%s" --export-pdf="%s.pdf"' %(options, file, pdf_abspath)) 


if __name__ == "__main__":
    print(__doc__)
    cwd, extension = getPaths()
    svg_files = getSVGList(cwd, extension)
    svg2pdf(svg_files)