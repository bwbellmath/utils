import pandas as pd
import numpy as np
from tabulate import tabulate
from PIL import Image
import os
import sys
import torch
from   torch.autograd import Variable
import torch.optim as optim 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
from   torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from   tqdm import *
from datetime import datetime, timedelta
import re
import gc
import json
import pdb
from timeit import default_timer as timer
from datetime import timedelta as timedelta
import git 

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
origin_url = repo.remotes["origin"].url
branch = repo.active_branch
origin_url = origin_url.replace("git@", "https://")
origin_url = origin_url.replace(":b", "/b")
#origin_url = origin_url.replace("git@github.com:bwbellmath", "https://github.com/bwbellmath")
#https://github.com/bwbellmath/stab/blob/main/img/stab-mnist-C32-100-100-10-0.001-0.0001-25-0.0001-dropout-db_xings-i-0-n_interp-1000.png
#https://github.com/bwbellmath/manifold-roubustness/blob/main/img/stab-mnist-C32-100-100-10-0.001-0.0001-25-0.0001-dropout-db_xings-i-3-n_interp-61000.png                                


# constants
HUGE = 1000000000
TEENY_WEENY = 0.000001
# constants
imsize = (224, 224)
n_iter = 30
n_real = 10
v_scal = 40


# updated summary.md generator -- old is now legacy!
class Report(object):
  def __init__(self, fo_md, fo_tex, first=True):
    self.fo_md = fo_md
    self.fo_tex = fo_tex
    self.first = first
    # read header file from repo
  
    with open("utils/latex_header.tex", "r") as file_header:
      self.header = file_header.readlines()


  # print document header for latex document 
  def summary_print(self, desc, desc_md, desc_tex):
    print("{}\n".format(desc))
    if (self.first):
      desc_head = self.header
      # make sure header is written first to latex file
      with open(self.fo_tex, "w") as f:
        f.writelines(desc_head)
        f.writelines(["\n"])
      #print("{}\n".format(desc_head), file=open(self.fo_tex,"w"))
      print("{}\n".format(desc_md), file=open(self.fo_md,"w"))
      print("{}\n".format(desc_tex), file=open(self.fo_tex,"a"))
      self.first = False
    else:
      print("{}\n".format(desc_md), file=open(self.fo_md,"a"))
      print("{}\n".format(desc_tex), file=open(self.fo_tex,"a"))

  def w_chapter(self, line):
    desc = line
    desc_md = "# {}".format(line)
    desc_tex = "\chapter{{ {} }}".format(line)
    self.summary_print(desc, desc_md, desc_tex)

  def w_section(self, line):
    desc = line
    desc_md = "# {}".format(line)
    desc_tex = "\section{{ {} }}".format(line)
    self.summary_print(desc, desc_md, desc_tex)

  def w_subsection(self, line):
    desc = line
    desc_md = "## {}".format(line)
    desc_tex = "\subsection{{ {} }}".format(line)
    self.summary_print(desc, desc_md, desc_tex)

  def w_subsubsection(self, line):
    desc = line
    desc_md = "\n ### {}".format(line)
    desc_tex = "\\ \n \subsubsection{{ {} }}".format(line)
    self.summary_print(desc, desc_md, desc_tex)

  def w_list(self, lines):
    desc = ""
    desc_tex = "\begin{enumerate}[1.] \n"
    desc_md = ""
    for l in lines:
      desc = desc+F"> {l} \n"
      desc_tex = desc_tex+F"\item {l} \n"
      desc_md = desc_md+F"1. {l} \n"
    self.summary_print(desc, desc_md, desc_tex)

  def w_lines(self, line):
    desc = line
    desc_md = line
    desc_tex = line
    self.summary_print(desc, desc_md, desc_tex)
  # TODO : put both image and table in a latex figure
  def w_image(self, line, f_img):
    # get url from git remote origin
    url = origin_url[0:-4]+"/blob/main/"

    desc = "{} \n Image URL : {}{}\n Image DIR : {} \n".format(line, url, f_img, f_img)
    desc_md = desc + "![{}]({}{})".format(f_img, url, f_img)
    desc_tex = "\includegraphics[width=0.9\\textwidth]{{{}}}".format(f_img)
    self.summary_print(desc, desc_md, desc_tex)

  def w_table(self, line, df_table):
    desc = line
    desc_md = line + "\n" + tabulate(df_table, tablefmt="pipe", headers="keys") + "\n"    
    desc_tex = line + "\n" + tabulate(df_table, tablefmt="latex", headers="keys") + "\n"
    self.summary_print(desc, desc_md, desc_tex)
  # make sure to do this last
  def w_tail(self):
    tail = "\end{document}"
    print(tail, file=open(self.fo_tex,"a"))

# summary.md generator
class Summary(object):
  def __init__(self, fo):
    self.fo = fo
    self.first = True
  # TODO : convert if statements and printing to generic function
  # TODO : have each summary function call generic print function
  # write description to covid_summary.md
  def summary_print(self, desc):
    print("{}\n".format(desc))
    if (self.first):
      print("{}\n".format(desc), file=open(self.fo,"w"))
      self.first = False
    else:
      print("{}\n".format(desc), file=open(self.fo,"a"))

  # generically write with no markdown formatting
  def wr(self, desc):
    descr = "{}".format(desc)
    self.summary_print(descr)
  def wd(self, desc):
    descr = "## {}".format(desc)
    self.summary_print(descr)
  # write title to covid_summary.md
  def wt(self, desc):
    descr = "# {}".format(desc)
    self.summary_print(descr)
  # write description with value to covid_summary.md
  def wdv (self, desc, data):
    descr = "### {} : {}".format(desc, data)
    self.summary_print(descr)
  def wi (self, fo):
    url = "https://github.com/bwbellmath/stab/blob/main/"
    descr = "### Included Image: {}".format(fo)
    self.summary_print(descr)
    descr_image = "![{}]({}{})".format(fo, url, fo)
    self.summary_print(descr_image)

# # summary.md generator
# class Summary(object):
#   def __init__(self, fo):
#     self.fo = fo
#     self.first = True
#   # TODO : convert if statements and printing to generic function
#   # TODO : have each summary function call generic print function
#   # write description to covid_summary.md
#   def summary_print(self, desc):
#     print("{}\n".format(desc))
#     if (self.first):
#       print("{}\n".format(desc), file=open(self.fo,"w"))
#       self.first = False
#     else:
#       print("{}\n".format(desc), file=open(self.fo,"a"))

#   def wd(self, desc):
#     descr = "## {}".format(desc)
#     self.summary_print(descr)
#     # print("## {}\n".format(desc))
#     # if (self.first):
#     #   print("## {}\n".format(desc), file=open(self.fo,"w"))
#     #   self.first = False
#     # else:
#     #   print("## {}\n".format(desc), file=open(self.fo,"a"))
#   # write title to covid_summary.md
#   def wt(self, desc):
#     descr = "# {}".format(desc)
#     self.summary_print(descr)
#     # print("# {}\n".format(desc))
#     # if (self.first):
#     #   print("# {}\n".format(desc), file=open(self.fo,"w"))
#     #   self.first = False
#     # else:
#     #   print("# {}\n".format(desc), file=open(fo,"a"))
#   # write description with value to covid_summary.md
#   def wdv (self, desc, data):
#     descr = "### {} : {}".format(desc, data)
#     self.summary_print(descr)
#     # print("### {} : {}\n".format(desc, data))
#     # if (self.first):
#     #   print("### {} : {}\n".format(desc, data), file=open(self.fo,"w"))
#     #   self.first = False
#     # else:
#     #   print("### {} : {}\n".format(desc, data), file=open(self.fo,"a"))


# examples

#  sw.wd("count of Patients(subject_id) that have each label")
#  sw.wdv("Reading file:", fil)
# print(tabulate(subject_count, tablefmt="pipe", headers="keys"),
#   file=open(sfo, "a"))
# print("\n", file=open(sfo, "a"))


# Add Beamer generator
class Beamer(object):
  def __init__(self,out_name,out_dir='./',title='',author=''):
    self.title = title
    self.author = author

    self.out_dir = out_dir
    self.out_name = out_name
    self.out_path = os.path.join(out_dir,out_name)
    # make out_dir only if it doesn't exist yet
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)

    # preamble
    self.text = '\\documentclass{beamer}\n\\usepackage{graphics}\n'
    self.text += '\\usetheme{Montpellier}\n'
    self.text += '\\usepackage[utf8]{inputenc}\n\\title{'+self.title+'}\n\\author{'+self.author+'}\n'
    self.text += '\\institute{University of Arizona}\n\\date{\\today}\n\n\\begin{document}\n'
    self.text += '\\titlepage{}\n'

  # function to start new frame
  def new_frame(self):
    self.text += '\\begin{frame}\n'

  # function to end frame
  def end_frame(self):
    self.text += '\\end{frame}\n\n'

  # function to add image
  def include_graphics(self,image,caption=''):
    self.text += '\\begin{figure}[h]\n'
    self.text += '\\includegraphics[width=.75\\textwidth,height=.75\\textheight]{'
    self.text += image + '}\n'
    self.text += '\\caption{'+caption+'}\n'
    self.text += '\\end{figure}\n'

  # function to end document and produce output
  def end(self):
    self.text += '\\end{document}'
    # create .tex file
    print(self.text,file=open(self.out_path+'.tex','w'))
    # compile
    os.system('pdflatex {}.tex -output-directory {}'.format(self.out_path, self.out_dir))
    # move auxilary files
    for ext in ['.aux','.nav','.pdf','.toc','.log','.out','.snm','.log']:
      if os.path.isfile('./'+self.out_name+ext):
        os.rename('./'+self.out_name+ext,self.out_path+ext)

class Directory(object):
  ddir = ""
  odir = ""
  def __init__(self):
    # TODO : put this information in a text or xml file for configuration
    # Sort out Directories
    brian_file = "C:/Users/Nexus/Desktop/stab_data/path_flag.txt"
    brian_desk = "C:/Users/brian/Desktop/stab_data/path_flag.txt"
    brian_laptop = "C:/Users/DuxLiteratum/Desktop/stab_data/path_flag.txt"
    brian_hpc = "/home/u22/bwbell/stab_data/path_flag.txt"
    brian_mbp = "/Users/brianbell/code/stab_data/path_flag.txt"
    brian_llnl = "/usr/workspace/bell73/stab_data/path_flag.txt"
    tyler_file = ""
    # please add "path_flag.txt" to the directory where you keep the data
    # also add your data path here for ddir (data sources) and odir (output location)
    if (os.path.isfile(brian_file)):
        self.ddir = "C:/Users/Nexus/Desktop/stab_data/"
        self.odir = "C:/Users/Nexus/Desktop/stab_data/"
    if (os.path.isfile(brian_file)):
        self.ddir = "C:/Users/Nexus/Desktop/stab_data/"
        self.odir = "C:/Users/Nexus/Desktop/stab_data/"
    elif (os.path.isfile(brian_laptop)):
        self.ddir = "C:/Users/DuxLiteratum/Desktop/stab_data/"
        self.odir = "C:/Users/DuxLiteratum/Desktop/stab_data/"
    elif (os.path.isfile(brian_hpc)):
        self.ddir = "/groups/dglicken/stab_data/"
        self.odir = "/groups/dglicken/stab_data/"
    elif (os.path.isfile(brian_mbp)):
        self.ddir = "/Users/brianbell/code/stab_data/"
        self.odir = "/Users/brianbell/code/stab_data/"
    elif (os.path.isfile(brian_llnl)):
        self.ddir = "/usr/workspace/bell73/stab_data/"
        self.odir = "/usr/workspace/bell73/stab_data/"
    elif (os.path.isfile(tyler_file)):
        self.ddir = ""
        self.ddir = ""
    else:
        raise SystemExit('Error: Cannot find ddir.')

def relabelTests_ce(df):

    #convert weights in lbs to kg (will relabel all as "weight" further down)
    df.loc[df.TEST=='Weight (lb)',['RESULT_VAL']]=pd.to_numeric(df.RESULT_VAL[df.TEST=='Weight (lb)'], errors='coerce')
    df.loc[df.TEST=='Weight (lb)',['RESULT_VAL']]=df.RESULT_VAL[df.TEST=='Weight (lb)']*0.4536
    df.loc[df.TEST=='Weight (lb)',['UNITS']]='Kilogram'

    #convert heights in inches to cm (will relabel all as "height" further down)
    df.loc[df.TEST.isin(['Height Calculation','Height (in)']),['RESULT_VAL']]=pd.to_numeric(df.RESULT_VAL[df.TEST.isin(['Height Calculation','Height (in)'])], errors='coerce')
    df.loc[df.TEST.isin(['Height Calculation','Height (in)']),['RESULT_VAL']]=df.RESULT_VAL[df.TEST.isin(['Height Calculation','Height (in)'])]*2.54
    df.loc[df.TEST.isin(['Height Calculation','Height (in)']),['UNITS']]='Centimeter'

    labels = {
    'Chloride':'chloride',
    'Respiratory Rate': 'RR',
    'Coronavirus (COVID-19) SARS-CoV-2 RNA':'COVID_pcr',
    'Glucose Level': 'glucose',
    'Systolic Blood Pressure' : 'SBP',
    'Diastolic Blood Pressure' : 'DBP',
    'Heart Rate Monitored' : 'HR',
    'Height (in)':  'height',
    'Bilirubin, UR': 'biliUr',
    'BUN/Creat Ratio': 'bun/Cr',
    'Oxygen Therapy': 'O2device',
    'Temperature (F)': 'temp',
    'Weight':'weight',
    'Weight (lb)':'weight',
    'BMI (Pt Care)' : 'BMI',
    'Glasgow Coma Score': 'GCS',
    'Heart Rate EKG' : 'HR',
    'Lymphocytes %': 'lymph%',
    'Bilirubin Total' : 'biliT',
    'NT-proBNP' : 'BNP',
    'Lymphocytes #': 'lymph#',
    'D-Dimer, Quant': 'd-dimer',
    'ED Document Glasgow Coma Scale' : 'GCS',
    'Platelet': 'platelets',
    'C Reactive Protein' : 'CRP',
    'Ferritin Level': 'Ferritin',
    'eGFR (non-African Descent)': 'GFR',
    'O2 Sat, ABG POC' : 'O2satABG',
    'D-Dimer Quantitative' : 'd-dimer',
    'Troponin-T, High Sensitivity': 'tropT_hs',
    'Troponin T, High Sensitivity': 'tropT_hs',
    'Cholesterol/HDL Ratio' : 'cholesterol/HDL',
    'Bilirubin  Direct' : 'biliD',
    'Height Calculation' : 'height',
    'Non HDL Cholesterol': 'chol_NonHDL',
    'Inspiratory Time': 'vent_iTime',
    'Transcribed Height (cm)' : 'height',
    'CRP High Sens' : 'CRP_hs',
    'MSOFA Score' : 'mSOFA',
    'FIO2, Arterial POC':'FiO2_abg',
    'HCO3(Bicarb), ABG POC': 'bicarb',
    'Peak Inspiratory Pressure': 'vent_PIP',
    'Coronavirus(COVID-19)SARS CoV2 TL Result' : 'COVID_tl',
    'Ventilator Mode': 'vent_mode',
    'Ventilator Frequency, Mandatory': 'vent_RRset',
    'Inspiratory to Expiratory Ratio': 'vent_IE',
    'Positive End Expiratory Pressure': 'vent_PEEP',
    'Auto-PEEP': 'vent_autoPEEP',
    'End Expiratory Pressure' : 'vent_PEEP',  #need to reassess O2 device that goes with this one to be sure it is vent
    'PCO2, ABG POC':'pCO2',
    'Inspiratory Flow Rate' : 'vent_iFlow',
    'Bilirubin Direct Serum' : 'biliD',
    'Inspiratory Pressure' : 'vent_inspP', #need to check this one goes with vent too
    'Inspiratory to Expiratory Ratio Measured' : 'vent_IE',
    'Interleukin-2 (IL-2)' : 'IL-2',
    'Interleukin-2 (IL-2) (RL)' : 'IL-2'}

    #replace labels
    df.replace(labels,inplace=True)

    return df



def relabelLabs_labs(df):
  #define shortened lab labels
  lablabels = {
    'Creatinine':'creatinine',
    'Monocytes %':'monocyte%',
    'Chloride':'chloride',
    'Monocytes #': 'monocyte#',
    'Anion Gap':'anionGap',
    'Basophils #':'basophils#',
    'Lymphocytes %':'lymph%',
    'Albumin':'albumin',
    'Lymphocytes #':'lymph#',
    'Metamyelocytes #': 'metamyelocytes#',
    'Myelocytes #': 'myelocytes#',
    'Basophils %':'basophils%',
    'Lipase Level':'lipase',
    'Hemoglobin A1c':'A1C',
    'Non HDL Cholesterol':'chol_NonHDL',
    'Beta-Hydroxybutyrate':'betaHydroxybutyrate',
    'Salicylate Level':'salicylateLevel',
    'LDL, Calculation':'LDL_calc',
    'Promyelocytes #': 'promyelocytes#',
    'Myelocytes %':'myelocytes%',
    'Retic #':'retic#',
    'Cholesterol/HDL Ratio': 'cholesterol/HDL',
    'Exp date':'expDate',
    'Ammonia':'amonia',
    'Protein C Activity':'proteinC_activity',
    'Metamyelocytes %': 'metamyelocytes%',
    'Acetaminophen Level':'acetaminophen_level',
    'Cardiolipin IgA Antibody':'cardiolipin_IgA',
    'Prealbumin':'prealbumin',
    'Hep B Surface Ab Result':'hepBs_Ab',
    'Promyelocytes %': 'promyelocytes%',
    'Vitamin B12 Level':'vitB12',
    'C. pneumoniae IgA': 'c_pnaIgA',
    'Vitamin D, 1, 25 (OH) Total': 'vitaminD',
    'Cardiolipin IgM Antibody':'cardiolipin_IgM',
    'Rheumatoid Factor':'rheumatoidFactor',
    'Activated Clotting Time POC': 'ACT',
    'Metanephrine, Free': 'metanephrine_free',
    'Plasma Hemoglobin': 'HGB',
    'Other Cells %':'otherCells%',
    'Basophils, Body Fluid %':'basophils_Fluid%',
    'Dev lot number': 'dev_Lot#',
    'Card lot number': 'card_Lot#',
    'C. pneumoniae IgG': 'c_pnaIgG',
    'Alpha-1-Antitrypsin': 'alpha1-antitrypsin',
    'Renin Activity, Plasma': 'renin-activity_plasma',
    'Specific Gravity, BF': 'specificGravity',
    'Complement, Total (CH50)': 'complementTot',
    'Cardiolipin IgG Antibody': 'cardiolipin_IgG',
    'Aldosterone': 'aldosterone',
    'HCV RNA, PCR, Quant (IU/mL)': 'hepC_RNA',
    'Other Cells #': 'otherCells#',
    'HCV RNA, PCR, Quant (LogIU/mL)':'hepCRNA_log',
    'Beta-2-Microglobulin':'Beta-2-Microglobulin',
    'Normetanephrine, Free':'normetanephrine_free',
    'PTH, Related Protein':'PTHrP',
    'Aldolase':'aldolase',
    'Results Reported To':'resultsCalledTo',
    'Time Frozen Result Called':'timeFrozenResultCalled',
    'Performing Pathologist':'pathologist',
    'Beta2- Glycoprotein 1 Ab IgA':'beta2Glycoprotein1_IgA',
    'Coxsackie A9 Ab':'coxsackieA9_Ab',
    'Esoteric Test Name':'esotericTest',
    'Prolactin':'prolactin',
    'Alkaline Phosphatase, Bone Specific':'alkphos_bone',
    'Beta2- Glycoprotein 1 Ab IgM': 'beta2Glycoprotein1_IgM',
    'Coxsackie A4 Ab': 'coxsackieA4_Ab',
    'Platelet Count, Citrated': 'platelets',
    'ACTH':'ACTH',
    'Normetanephrine':'normetanephrine',
    'Metanephrine':'metanephrine',
    'Vitamin B6 level':'vitaminB6',
    'Metanephrines, Total':'metanephrine_tot'}

  df.replace(lablabels, inplace=True)

  return df



def relabelO2(df):
  #define non-redundant O2 delivery device types
  #we will use types roomAir, NC (nasal cannula), hiFlowNC (highFlowNC flow nasal cannula),
  #  openMask (any venti mask, face tent, simple mask, etc), nonRebreather, NIPPV (cpap/bipap),
  #  BVM (bag valve mask), vent (ventilator), trach mask, and t-piece
  #When two devices ar listed, we will take the one with greater O2 delivery or greater invasiveness
    O2devices = {
    'Room air':'roomAir',

    'Nasal cannula':'NC',
    'Room air, Nasal cannula':'NC',
    'Humidification, Nasal cannula':'NC',

    'High-Flow nasal cannula':'hiFlowNC',
    'High-Flow nasal cannula, Humidification':'hiFlowNC',
    'High-Flow nasal cannula, Nasal cannula':'hiFlowNC',
    'High-Flow nasal cannula, Humidification, Nasal cannula':'hiFlowNC',
    'High-Flow nasal cannula, Venti-mask':'hiFlowNC',
    'High-Flow nasal cannula, Oxymask':'hiFlowNC',
    'High-Flow nasal cannula, Humidification, Venti-mask':'hiFlowNC',

    'Aerosol mask':'openMask',
    'Simple mask':'openMask',
    'Oxymask':'openMask',
    'Room air, Venti-mask':'openMask',
    'Venti-mask':'openMask',
    'Humidification':'openMask',
    'Room air, Simple mask':'openMask',
    'Nasal cannula, Venti-mask':'openMask',
    'Blow-By':'openMask',
    'Mist tent':'openMask',
    'Nasal cannula, Simple mask':'openMask',
    'Aerosol mask, Humidification':'openMask',
    'Face shield':'openMask',
    'Simple mask, Venti-mask':'openMask',
    'Face shield, Nasal cannula':'openMask',
    'Humidification, Simple mask':'openMask',
    'Nasal cannula, Oxymask':'openMask',

    'Nonrebreather mask':'nonRebreather',
    'High-Flow nasal cannula, Nonrebreather mask':'nonRebreather',
    'High-Flow nasal cannula, Nonrebreather mask, Venti-mask':'nonRebreather',
    'High-Flow nasal cannula, Humidification, Nonrebreather mask':'nonRebreather',
    'Partial rebreather mask':'nonRebreather',
    'Humidification, Nasal cannula, Nonrebreather mask':'nonRebreather',
    'Nasal cannula, Nonrebreather mask':'nonRebreather',
    'Humidification, Nasal cannula, Simple mask':'nonRebreather',
    'Room air, Nonrebreather mask':'nonRebreather',
    'Nonrebreather mask, Partial rebreather mask':'nonRebreather',
    'Nasal cannula, Nonrebreather mask, Venti-mask':'nonRebreather',
    'Nonrebreather mask, Venti-mask':'nonRebreather',

    'CPAP':'NIPPV',
    'BiPAP':'NIPPV',
    'BiPAP, Venti-mask':'NIPPV',
    'CPAP, Venti-mask':'NIPPV',
    'Room air, BiPAP':'NIPPV',

    'Bag valve mask':'BVM',

    'Ventilator':'vent',
    'Humidification, Ventilator':'vent',
    'CPAP, Ventilator':'vent',
    'Trach shield, Ventilator':'vent',

    'Trach shield':'trachMask',
    'Transtracheal (TTO)':'trachMask',
    'Trach shield, Transtracheal (TTO)': 'trachMask',
    'Aerosol mask, Blow-By, Simple mask, Transtracheal (TTO)':'trachMask',

    'Room air, T-piece':'t-piece',
    'T-piece':'t-piece',

    #for a few of these where the 2 listed devices are very different (vent vs not, trach vs not)
    #  we will preserve the ambiguity for now
    'High-Flow nasal cannula, Transtracheal (TTO)':'trachMask vs hiFlowNC',
    'Nasal cannula, Transtracheal (TTO)':'trachMask vs NC',
    'Nasal cannula, Ventilator':'vent vs NC',
    'Ventilator, Venti-mask':'vent vs openMask',
    'Simple mask, Ventilator':'vent vs openMask',
    'Room air, Ventilator':'vent vs roomAir',
    'High-Flow nasal cannula, Ventilator':'vent vs hiFlowNC',
    'Nonrebreather mask, Ventilator':'vent vs nonRebreather',
    'High-Flow nasal cannula, Ventilator, Venti-mask':'vent vs hiFlowNC'}

    df.replace(O2devices, inplace=True)
    return df



# function to obtain dictionaries for numaric and nonnumeric values for labels
##Input dataframe, col1 is the name of the column that gives the test names, col2 is the name of the column with the values
def nonnumeric(dd, col1, col2):
    unique_tests = pd.unique(dd[col1])

    numvals = dict()
    nonnumvals = dict()
    for test in unique_tests:
        sid = dd[col1] == test
        dis = dd[sid]

        unique_vals = pd.unique(dis[col2])

        digitvals = []
        nondigitvals = []
        for v in unique_vals:
            try:
                float(v)
                digitvals.append(v)

            except ValueError:
                nondigitvals.append(v)

        numvals[test] = digitvals
        nonnumvals[test] = nondigitvals

    return numvals, nonnumvals


# function clean fio2 vals
def clean_fio2_val(text):
  # if str, look for numeric string
  if type(text) is str:
    # if no digits, return nan
    exp = re.compile('\d')
    match = re.findall(exp,text)
    if len(match) == 0:
      return np.nan
    # if there are digits, convert to float
    # multiply by 0.01 to convert from pct if applicable
    if '%' in text:
      pct_fact = 0.01
    else:
      pct_fact = 1
    # convert from lpm if applicable
    if 'lpm' in text:
      lpm_flag=True
    else:
      lpm_flag=False
    exp = re.compile('[^0-9\.]')
    new = float(re.sub(exp,'',text))
    if lpm_flag:
      new = 0.2+0.04*new
    return new * pct_fact
  # if float, do nothing
  elif type(text) is float:
    return text
  # if int, make float
  elif type(text) is int:
    return float(text)
  # if type not found, return warning and nan
  else:
    print('Warning: {} type {} not supported. Returning nan'.format(text,type(text)))
    return np.nan

# function to clean fio2 column
def clean_fio2_col(column):
  # if series, convert to list
  if type(column) is pd.core.series.Series:
    column = column.values
  # initialize array of indices to drop
  replace = np.zeros(len(column))
  # loop
  for idx, val in enumerate(column):
    # clean val, make replace array if nan
    new = clean_fio2_val(val)
    column[idx] = new
    if np.isnan(new):
      replace[idx] = 1
  return column, replace


# function to clean column of miscellaneous string values
def clean_col(column):
  # if series, convert to list
  if type(column) is pd.core.series.Series:
    column = column.values
  # initialize array of indices to drop
  replace = np.zeros(len(column))
  # loop
  for idx, val in enumerate(column):
    # clean val, make replace array if nan
    new = clean_string(val)
    column[idx] = new
    if np.isnan(new):
      replace[idx] = 1
  return column, replace

# function to remove inequality
def clean_string(val):

  # if str, clean
  if type(val) is str:
    # purely numeric, return as float
    exp = re.compile('[^0-9\.]')
    match = re.findall(exp,val)
    if len(match) == 0:
      return float(val)
    # if no ineq, return orig, otherwise remove and convert to float
    exp = re.compile('[\<\>]')
    match = re.findall(exp,val)
    if len(match) > 0:
      new = float(re.sub(exp,'',val))
      return new

    # if neither apply, then replace with nan
    return np.nan
  # don't change if it's not a string
  else:
    return val

  
# function to clean column of ratios
def clean_ratio_col(column):
  # if series, convert to list
  if type(column) is pd.core.series.Series:
    column = column.values
  # initialize array of indices to drop
  replace = np.zeros(len(column))
  # loop
  for idx, val in enumerate(column):
    # clean val, make replace array if nan
    new = clean_ratio(val)
    column[idx] = new
    if np.isnan(new):
      replace[idx] = 1
  return column, replace

# function to fix ratios
def clean_ratio(val):
  # if str, clean
  if type(val) is str:
    # purely numeric, return as float
    exp = re.compile('[^0-9\.]')
    match = re.findall(exp,val)
    if len(match) == 0:
      try:
        return float(val)
      except:
        return np.nan
    # if : is there, compute decimal
    exp = re.compile('([0-9\.]*)\:([0-9\.]*)')
    match = re.search(exp,val)
    if match:
      # try to return as decimal, if doesn't work, return nan
      try:
        return float(match[1]) / float(match[2])
      except:
        return np.nan
    
    # if neither apply, then replace with nan
    return np.nan
  # don't change if it's not a string
  else:
    return val

# function to clean column of positive/negatives
def clean_posneg_col(column):
  # if series, convert to list
  if type(column) is pd.core.series.Series:
    column = column.values
  # initialize array of indices to drop
  replace = np.zeros(len(column))
  # loop
  for idx, val in enumerate(column):
    # clean val, make replace array if nan
    new = clean_posneg(val)
    column[idx] = new
    if np.isnan(new):
      replace[idx] = 1
  return column, replace

# function to fix positive/negative
def clean_posneg(val):
  # if str, clean
  if type(val) is str:
    # purely numeric, return as float
    exp = re.compile('[^0-9\.]')
    match = re.findall(exp,val)
    if len(match) == 0:
      try:
        return float(val)
      except:
        return np.nan
    # Possible strings for 1
    pos_strings = ['Detected','DETECTED','Positive']
    neg_strings = ['Not Detected','Not detected','NOT DETECTED','Negative']

    if val in pos_strings:
      return 1.0

    elif val in neg_strings:
      return 0.0

    else:
      return np.nan
    
  else:
    return val


# TODO : Add labeling function for which patients got intubated.

# TODO : Add code which analyzes text fields and replaces them with
#        numeric values (should have a dictionary of terms and
#        replacements.
#       -should probably read all of the input files, fix them, and
#        output "cleaned" input files so that we don't need to do that
#        every time.

def generate_dict(column, fo):
  dictionary = {}
  # get unique string values
  # for each unique string value
  # col_dict[string] = {}
  # col_dict[string]["original"] = string
  # col_dict[string]["suggested"] = clean_val(string)
  # col_dict[string]["replacement"] = ""
  # json.dump(col_dict, open(fo, 'w'))

class Cleaner(object):
  label_dict = {}
  value_dict = {}
  def __init__(self, ddir):
    label_file = ddir+"label_dict.txt"
    value_file = ddir+"value_dict.txt"
    self.label_dict = json.load(open(label_file))
    self.value_dict = json.load(open(value_file))


# functions
def flat_trans(x):
    x.resize_(28 * 28)
    return x

loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = image.convert("RGB") # Auto remove the "alpha" channel from png image
    image = loader(image).float()
    image = normalize(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image 

def n_normalize(image_data):
  c = image_data
  #e = (c+np.abs(np.min(c)))
  e = np.abs(c+c.min())
  #f_x = e/np.max(np.abs(e))
  f_x = e/np.abs(e.max())
  return f_x

def n_box (din_x, din_n):
    a = din > 1.0
    b = din < 0.0
    din[a] = 1.0
    din[b] = 0.0

# function that adds a_value to adict[alist[0]][alist[1]][...
def add_dict(alist, adict, avalue):
    if (len(alist) > 1):
        if (alist[0] not in adict):
            adict[alist[0]] = {}
        adict[alist[0]] = add_dict(alist[1:], adict[alist[0]], avalue)
        return adict
    else:
        adict[alist[0]] = avalue
        return adict

def check_dict(alist, adict):
    if (len(alist) > 1):
        if (alist[0] not in adict):
            return False
        else:
            return check_dict(alist[1:], adict)
    else:
        return alist[0] in adict


# define bracketing function
def imnet_bracketting (image, net, n, tol, n_real):
		# start with same magnitude noise as image
  a_var0 = np.var(x.detach().numpy())/4
  a_var = a_var0
  l_var = 0
  u_var = a_var*2
  a_vars = np.zeros(n)
  # Adversarial image plus noise counts
  a_counts = np.zeros(n)
  count = 0
  rap = np.zeros([n_real,n_iter])
		# grab the classification of the image under the network
  with torch.no_grad():
    y_a = np.argmax(net.forward(Variable(image)).data.numpy())
  print("Original Classification:\t({}) {}".format(
         y_a,
         ImageNet_mapping[str(y_a)]))
  # check that u_var is high enough
  print("Is u_var is high enough?")
  
  m = np.zeros([n_real,3,224,224])
  samp = np.random.normal(0,np.sqrt(u_var),m.shape)
  samp_t = Variable(torch.Tensor(samp))
  pert_a = image.data + samp_t
  start = timer()
  with torch.no_grad():
    image_a = net.forward(Variable(pert_a)).data.numpy()
  end = timer()
  image_as = np.argmax(image_a,axis=1)
  print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
  while(np.sum(image_as == y_a) > n_real*tol/2):
    u_var = u_var*2
    samp = np.random.normal(0,np.sqrt(u_var),m.shape)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image.data + samp_t
    start = timer()
    with torch.no_grad():
      image_a = net.forward(Variable(pert_a)).data.numpy()
    end = timer()
    image_as = np.argmax(image_a,axis=1)
    print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
    
 
  # perform the bracketing 
  for i in range(0,n): #, a_var in enumerate(a_vars):
    count+=1
    print("Starting iteration: {}, sample variance: {}".format(count, a_var))
  		# compute sample and its torch tensor
    samp = np.random.normal(0,np.sqrt(a_var),m.shape)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image.data + samp_t
    start = timer()
    with torch.no_grad():
      image_a = net.forward(Variable(pert_a)).data.numpy()
    end = timer()
    print("Time For {}: {}".format(n_real,timedelta(seconds=end-start)))
    image_as = np.argmax(image_a,axis=1)
    rap[:,i] = image_as

    a_counts[i] = np.sum(image_as == y_a)
    a_vars[i] = a_var # save old variance

    print("count:{}, interval: [{},{}]".format(a_counts[i], l_var, u_var))
		#floor and ceiling surround number
    if ((a_counts[i] <= np.ceil(n_real*tol)) & (a_counts[i] > np.floor(n_real*tol))):
        return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}        
    elif (a_counts[i] < n_real*tol): # we're too high
        print("We're too high, going from {} to {}".format(a_var, (a_var+l_var)/2)) 
        u_var = a_var
        a_var = (a_var + l_var)/2
    elif (a_counts[i] >= n_real*tol): # we're too low
        l_var = a_var
        print("We're too low,  going from {} to {}".format(a_var,(u_var+a_var)/2))
        a_var = (u_var + a_var)/2
        #u_var = u_var*2

  return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}


def sampling(image, net, n, n_real, s_min, s_max):
  variances = np.linspace(s_min, s_max, n)
  n_sz = np.array(image.shape)
  n_sz[0] = n_real
  image = image.cpu()

  if torch.cuda.is_available():
    net.to('cuda')
    
  # for mnist -- make sure image shape has 4 dimensions (#, channels, L, W)

  i = 0
  rap = np.zeros([n_real,n])
  ishape = np.array(image.shape)
  ishape[0] = n
  image_samples = np.zeros(ishape)
  for var in variances:
    # generate gaussian sample with that var
    print("Generating sample {} for var: {}".format(i, var))
    samp = np.random.normal(0,np.sqrt(var),n_sz)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image + samp_t
    if torch.cuda.is_available():
      pert_a = pert_a.to("cuda")
      
    with torch.no_grad():
      image_a = net.forward(Variable(pert_a.float())).data.cpu().numpy()
    #end = timer()
    # TODO : persistance to label class
    #print("Time For {}: {}".format(n_real,timedelta(seconds=end-start)))
    image_as = np.argmax(image_a,axis=1)
    rap[:,i] = image_as
    image_samples[i] = pert_a.cpu()[0]
    # record sample images
    # don't forget to increment index
    i+=1
  odict = {}
  odict["variances"] = variances
  odict["class_counts"] = rap
  odict["image_samples"] = image_samples
  return odict

def sampline(image_1, image_2, net, n, n_real, sigma):
  variances = np.linspace(0.0, 1.0, n)
  n_sz = np.array(image_1.shape)
  n_sz[0] = n_real
  image_1 = image_1.cpu()
  image_2 = image_2.cpu()  

  if torch.cuda.is_available():
    net.to('cuda')

  # for mnist -- make sure image shape has 4 dimensions (#, channels, L, W)

  i = 0
  rap = np.zeros([n_real,n])
  ishape = np.array(image_1.shape)
  ishape[0] = n
  image_samples = np.zeros(ishape)
  diff = image_2 - image_1
  for var in variances:
    # generate gaussian sample with that var
    print("Generating sample {} for step: {}".format(i, var))
    samp = np.random.normal(0,np.sqrt(sigma),n_sz)
    samp_t = Variable(torch.Tensor(samp))
    pert_a = image_1 + var*(diff) + samp_t
    if torch.cuda.is_available():
      pert_a = pert_a.to("cuda")
      
    with torch.no_grad():
      image_a = net.forward(Variable(pert_a.float())).data.cpu().numpy()
    #end = timer()
    # TODO : persistance to label class
    #print("Time For {}: {}".format(n_real,timedelta(seconds=end-start)))
    image_as = np.argmax(image_a,axis=1)
    rap[:,i] = image_as
    image_samples[i] = pert_a.cpu()[0]
    # record sample images
    # don't forget to increment index
    i+=1
  odict = {}
  odict["variances"] = variances
  odict["class_counts"] = rap
  odict["image_samples"] = image_samples
  return odict


def samplot(fo, class_counts, image_samples, variances, c_l, c_i, c_a, clss):
  ishape = list(image_samples[0].shape)
  ishape[-1]*=5
  i_disp = np.zeros(ishape)
  # grab 5 image samples
  icount = len(image_samples)
  for i in range(0,5):
    ind = int(np.floor(icount/5*i))
    i_disp[:,0*ishape[-2]:1*ishape[-2], i*ishape[-2]:(i+1)*ishape[-2]] = image_samples[ind]
    # maybe need to swap axes?
  #DG next two lines
  fonts = 16  # Font size for legend and title
  plt.rcParams.update({'font.size': fonts})

  # compute counts for each class and variance
  dcounts = {}
  for index in np.unique(class_counts):
    dcounts[int(index)] = np.zeros(class_counts.shape[1])
  for ii in range(class_counts.shape[1]):
    a, b = np.unique(class_counts[:,ii], return_counts=True)
    for jj in range(len(a)):
      #print("iter : {} -- adding count: {} for class {}".format(ii, b[jj], a[jj]))
      dcounts[a[jj]][ii] = b[jj]

  # create figure with subplots
  fig, ax= plt.subplots(2)#,sharex=True, sharey=True)
  # if adversarial:
  if (c_a < 1000):
    plt.suptitle("Adversarial Class: {} ({}), Original Class: {} ({})".format(clss[c_a], c_a, clss[c_i], c_i), fontsize=str(fonts))
  else:
    plt.suptitle("Original Class: {} ({})".format(clss[c_i], c_i), fontsize=str(fonts))
  # else:
  fig.set_size_inches(12,6)

  # plot -- color c_l, c_i, c_a as designated
  
  # 
  for key in dcounts.keys():
    # do this last
    lwidth = 1.0
    color = "blue"

    # '#999999' : label class (c_l)
    # black :  original predicted class (c_i) and
    # red : adversarial predictedclass  (c_a) 
    ax[0].plot(variances, dcounts[key], linewidth=lwidth, color=color, alpha=0.2)

  last_keys = [c_i, c_a]
  for key in last_keys:
    # if (key == c_l):
    #   lwidth = 3.0
    #   color = "#999999"
    #   # do this third to last
    if (key > 1000):
      continue
    if (key == c_i):
      lwidth = 3.0
      color = "black"
      # do this second to last
    elif (key == c_a):
      # make sure that if this is > 1000 i.e. a natural image, this is not drawn
      lwidth = 3.0
      color = "red"
      # do this last
    print("Plotting {} in color: {}".format(key, color))
    if (key in dcounts.keys()):
      ax[0].plot(variances, dcounts[key], linewidth=lwidth, color=color)
    else:
      ax[0].plot(variances, np.zeros(len(variances)), linewidth=lwidth, color=color)

  ax[0].set_ylabel('Count',labelpad=-5)
  ax[0].set_xlabel('Standard Deviation ($\sigma$)')  
  if (c_a > 1000):
    custom_lines = [Line2D([0], [0], color="black", lw=4),
                    #Line2D([0], [0], color= "#999999", lw=4),
                    Line2D([0], [0], color="blue", lw=4, alpha=0.2)]
    ax[0].legend(custom_lines, ["Orig Class", "Other Classes"], loc="upper right")
  else:
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="black", lw=4),
                    #Line2D([0], [0], color= "#999999", lw=4),
                    Line2D([0], [0], color="blue", lw=4, alpha=0.2)]
    ax[0].legend(custom_lines, ["Adv Class", "Orig Class", "Other Classes"], loc="upper right")


  # imshow i_disp
  ax[1].imshow(i_disp.swapaxes(0,2).swapaxes(0,1))#(a_sigs,results_a.T, linewidth=2.0)
  # remove plot axes for this part
  ax[1].get_xaxis().set_visible(False)
  ax[1].set_ylabel("Sample Images", labelpad=30)
  ax[1].yaxis.set_tick_params(length=0)
  #ax[1].get_xaxis().set_visible(False)
  #ax[1].get_yaxis().set_visible(False)


  # example
  #plt.gca().set_axis_off()
  plt.subplots_adjust(top = 0.93, bottom = 0, right = 0.99, left = 0.06, 
              hspace = 0.2, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  print("dumping to {}".format(fo))
  #fo = "/Users/brianbell/Pictures/samplot_test.png"
  # bbox_inches = 'tight', 
  plt.savefig(fo, pad_inches = 0)

def samplot_interp(fo, class_counts, image_samples, variances, c_l, c_i, c_a, clss, sigma):
  ishape = list(image_samples[0].shape)
  ishape[-1]*=5
  i_disp = np.zeros(ishape)
  # grab 5 image samples
  icount = len(image_samples)
  for i in range(0,5):
    ind = int(np.floor(icount/5*i))
    i_disp[:,0*ishape[-2]:1*ishape[-2], i*ishape[-2]:(i+1)*ishape[-2]] = image_samples[ind]
    # maybe need to swap axes?
  #DG next two lines
  fonts = 16  # Font size for legend and title
  plt.rcParams.update({'font.size': fonts})

  # compute counts for each class and variance
  dcounts = {}
  for index in np.unique(class_counts):
    dcounts[int(index)] = np.zeros(class_counts.shape[1])
  for ii in range(class_counts.shape[1]):
    a, b = np.unique(class_counts[:,ii], return_counts=True)
    for jj in range(len(a)):
      #print("iter : {} -- adding count: {} for class {}".format(ii, b[jj], a[jj]))
      dcounts[a[jj]][ii] = b[jj]

  # create figure with subplots
  fig, ax= plt.subplots(2)#,sharex=True, sharey=True)
  # if adversarial:
  if (c_a < 1000):
    plt.suptitle("$\sigma$={}: Adversarial Class: {} ({}), Original Class: {} ({})".format(sigma, clss[c_a], c_a, clss[c_i], c_i), fontsize=str(fonts))
  else:
    plt.suptitle("Original Class: {} ({})".format(clss[c_i], c_i), fontsize=str(fonts))
  # else:
  fig.set_size_inches(12,6)

  # plot -- color c_l, c_i, c_a as designated
  
  # 
  for key in dcounts.keys():
    # do this last
    lwidth = 1.0
    color = "blue"

    # '#999999' : label class (c_l)
    # black :  original predicted class (c_i) and
    # red : adversarial predictedclass  (c_a) 
    ax[0].plot(variances, dcounts[key], linewidth=lwidth, color=color, alpha=0.2)

  last_keys = [c_i, c_a]
  for key in last_keys:
    # if (key == c_l):
    #   lwidth = 3.0
    #   color = "#999999"
    #   # do this third to last
    if (key > 1000):
      continue
    if (key == c_i):
      lwidth = 3.0
      color = "black"
      # do this second to last
    elif (key == c_a):
      # make sure that if this is > 1000 i.e. a natural image, this is not drawn
      lwidth = 3.0
      color = "red"
      # do this last
    print("Plotting {} in color: {}".format(key, color))
    if (key in dcounts.keys()):
      ax[0].plot(variances, dcounts[key], linewidth=lwidth, color=color)
    else:
      ax[0].plot(variances, np.zeros(len(variances)), linewidth=lwidth, color=color)

  ax[0].set_ylabel('Count',labelpad=-5)
  ax[0].set_xlabel('Interpolation Progress')  
  if (c_a > 1000):
    custom_lines = [Line2D([0], [0], color="black", lw=4),
                    #Line2D([0], [0], color= "#999999", lw=4),
                    Line2D([0], [0], color="blue", lw=4, alpha=0.2)]
    ax[0].legend(custom_lines, ["Orig Class", "Other Classes"], loc="upper right")
  else:
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="black", lw=4),
                    #Line2D([0], [0], color= "#999999", lw=4),
                    Line2D([0], [0], color="blue", lw=4, alpha=0.2)]
    ax[0].legend(custom_lines, ["Adv Class", "Orig Class", "Other Classes"], loc="upper right")


  # imshow i_disp
  ax[1].imshow(i_disp.swapaxes(0,2).swapaxes(0,1))#(a_sigs,results_a.T, linewidth=2.0)
  # remove plot axes for this part
  ax[1].get_xaxis().set_visible(False)
  ax[1].set_ylabel("Sample Images", labelpad=30)
  ax[1].yaxis.set_tick_params(length=0)
  #ax[1].get_xaxis().set_visible(False)
  #ax[1].get_yaxis().set_visible(False)


  # example
  #plt.gca().set_axis_off()
  plt.subplots_adjust(top = 0.93, bottom = 0, right = 0.99, left = 0.06, 
              hspace = 0.2, wspace = 0)
  plt.margins(0,0)
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  print("dumping to {}".format(fo))
  #fo = "/Users/brianbell/Pictures/samplot_test.png"
  # bbox_inches = 'tight', 
  plt.savefig(fo, pad_inches = 0)

# compute persistence(
# convert this to do it with arrays of points instead
# TODO : rename to persistence and convert all cpu tensors to "device"
def persistence (image, net, n, gamma, n_real, c_i, tol):
		# start with same magnitude noise as image
  utol = 1.0+tol
  stol = 1.0-tol
  a_var0 = torch.var(image)#.detach().cpu().numpy())
  a_var = a_var0
  l_var = 0
  u_var = a_var*2
  a_vars = torch.zeros(n)
  # Adversarial image plus noise counts
  a_counts = torch.zeros(n)
  n_sz = torch.tensor(image.shape)
  n_sz[0] = n_real
  mean = torch.zeros(tuple(n_sz))


  count = 0
  rap = torch.zeros([n_real,n])
		# grab the classification of the image under the network
  if torch.cuda.is_available():
    net.to('cuda')
    image = image.to("cuda")


  with torch.no_grad():
    y_a = torch.argmax(net(image), axis=1)
    #image_a = net.forward(Variable(pert_a.float())).data.cpu().numpy()


  print("Original Classification:\tO: {} -> A: {}".format(
         int(c_i), y_a))
  #       ImageNet_mapping[str(y_a)]))
  # check that u_var is high enough
  #print("Is u_var is high enough?")
  
  #cov  = u_var*torch.identity(n_sz)
  #samp = torch.random.multivariate_normal(mean, cov, n_real)
  #samp = torch.random.normal(0,torch.sqrt(u_var),n_sz)
  samp = torch.normal(mean, 1)
  samp_t = samp*torch.sqrt(a_var)
  imgs = torch.ones(samp_t.shape)
  imgs[None,:,:,:] = image
  pert_a = imgs + samp_t#(image+samp_t).clone().detach().requires_grad_(True)# + 
  with torch.no_grad():
    image_a = net(pert_a) #net.forward(Variable(pert_a.float())).data.cpu().numpy()

  image_as = torch.argmax(image_a,axis=1)
  #print("u_var:{}, count:{}".format(u_var, torch.sum(image_as == y_a)))
  # expand the range
  while((image_as == y_a).sum() > n_real*gamma*stol): #n_real*tol/2):
    u_var = u_var*2
    #cov  = u_var*torch.identity(n_sz)
    #samp = torch.random.multivariate_normal(mean, cov, n_real)
    samp = torch.normal(mean, 1)
    samp_t = samp*torch.sqrt(u_var)
    imgs = torch.ones(samp_t.shape)
    imgs[None,:,:,:] = image
    pert_a = imgs + samp_t#(image+samp_t).clone().detach().requires_grad_(True)# + 
    with torch.no_grad():
      image_a = net(pert_a) #net.forward(Variable(pert_a.float())).data.cpu().numpy()

    image_as = torch.argmax(image_a,axis=1)
    print("u_var:{}, count:{}".format(u_var, (image_as == y_a).sum()))
    
 
  # perform the bracketing 
  for i in range(0,n): #, a_var in enumerate(a_vars):
    count+=1
    #print("Starting iteration: {}, sample variance: {}".format(count, a_var))
  		# compute sample and its torch tensor
    #cov  = u_var*torch.identity(n_sz)
    #samp = torch.random.multivariate_normal(mean, cov, n_real)
    samp = torch.normal(mean, 1)
    samp_t = samp*torch.sqrt(a_var)
    imgs = torch.ones(samp_t.shape)
    imgs[None,:,:,:] = image
    pert_a = imgs + samp_t#(image+samp_t).clone().detach().requires_grad_(True)# + 
    with torch.no_grad():
      image_a = net(pert_a) #net.forward(Variable(pert_a.float())).data.cpu().numpy()

    image_as = torch.argmax(image_a,axis=1)
    rap[:,i] = image_as

    a_counts[i] = (image_as == y_a).sum()
    a_vars[i] = a_var # save old variance

    #print("count:{}, interval: [{},{}]".format(a_counts[i], l_var, u_var))
		#floor and ceiling surround number
    if ((a_counts[i] <= torch.ceil(torch.tensor(n_real*gamma))) & (a_counts[i] > torch.floor(torch.tensor(n_real*gamma)))):
        results = dict()
        results["a_sigs"] = a_vars
        results["a_counts"] = a_counts
        results["a_var"] = a_var
        #results["results_o"] = results_o # np.zeros([10,n_iter])
        return(results)
        #return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}        
    elif (a_counts[i] < n_real*gamma): # we're too high
        #print("We're too high, going from {} to {}".format(a_var, (a_var+l_var)/2)) 
        u_var = a_var
        a_var = (a_var + l_var)/2
    elif (a_counts[i] >= n_real*gamma): # we're too low
        l_var = a_var
        #print("We're too low,  going from {} to {}".format(a_var,(u_var+a_var)/2))
        a_var = (u_var + a_var)/2
        #u_var = u_var*2

  #return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}
  results = dict()
  results["a_sigs"] = a_vars
  results["a_counts"] = a_counts
  results["a_var"] = a_var
  #results["results_o"] = results_o # np.zeros([10,n_iter])
  return(results)

# compute persistence(
# convert this to do it with arrays of points instead
def persistence_tensor (images, net, b_iter, gamma, n_real, tol, image_index):
		# start with same magnitude noise as image
  ii = image_index
  results = dict()
  results["All_images_within_tolerance"] = True
  bound_shape = images.shape[0]
  utol = torch.ones(bound_shape)+tol
  stol = torch.ones(bound_shape)-tol
  a_var0 = torch.var(images.flatten(start_dim=1), axis=1)
  a_var = a_var0
  l_var = torch.zeros(a_var0.shape)
  u_var = a_var*2
  a_vars = torch.zeros([b_iter, a_var0.shape[0]])
  # Adversarial image plus noise counts
  a_counts = torch.zeros([b_iter, a_var0.shape[0]])

  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device="cuda"
  else:
    device = "cpu"


  #rap = np.zeros([n_real,b_iter])
		# grab the classification of the image under the network	
  # net and images should already be on cu
  #if torch.cuda.is_available():
  #  net.to('cuda')
  #  images = images.to("cuda")
  images = images.reshape(images.shape)
  with torch.no_grad():
    y_a = torch.argmax(net(images), axis=1)
    #image_a = net.forward(Variable(pert_a.float())).data.cpu().numpy()

  print(F"Computing Persistence for imgage {ii} : \nLeft Pred: {y_a[0]}, \nRight Pred: {y_a[-1]}")
  #       ImageNet_mapping[str(y_a)]))
  # check that u_var is high enough
  #print("Is u_var is high enough?")
  
  #cov  = u_var*np.identity(n_sz)
  #samp = np.random.multivariate_normal(mean, cov, n_real)

  #samp = np.random.normal(0,1,n_sz)
  #multiply each of sample by sqrt(u_var)  
  
  #tart = timer()
  # flatten images before net

  b_size = int((1024*2*2*2)/n_real)
  # want each batch to be about 4000, so n_real must be less than 4000ish
  # b_size = 
  pert_loader = torch.utils.data.DataLoader(dataset=images, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
  pred = torch.zeros([images.shape[0], n_real])
  for j, data in enumerate(pert_loader, 0):
    gc.collect()
    n_sz = np.append(np.array(n_real), np.array(data.shape))
    mean = torch.zeros(tuple(n_sz))
    samp_t = torch.normal(mean, 1)
    # TODO : a_var subset must correspond with pre
    var_s = a_var[j*b_size:(j+1)*b_size]
    samp_t = samp_t*torch.sqrt(var_s)[:, None, None, None]
    #images_flat = images.flatten(start_dim = 1)
    #samp_t = Variable(torch.Tensor(samp))
    pert_a = data[None, :, :, :] + samp_t
    pert_af = pert_a.flatten(start_dim=0, end_dim=1)
    if torch.cuda.is_available():
      pert_af = pert_af.to("cuda")

    with torch.no_grad():
      ppred = torch.argmax(net(pert_af), axis=1)
      pred[j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)
    #pred[j*b_size:(j+1)*b_size] = ppred.detach().reshape(b_size, n_real)

    #pred = torch.argmax(net(pert_af), axis=1)#Variable(pert_af.float())).data.cpu().numpy()
  # unflatten prediction after
  # keep imshow plot pred_uf before, during, and after (3 total plots) when plot turned on
  pred_uf = pred #pred.reshape(tuple(pert_a.shape[0:2]))
  results["pred_heatmap_before"] = pred_uf
  #end = timer()
  #print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))

  # expand the range
  #cool plot : shows gamma across interpolation (pred_uf == y_a).sum(axis=0)
  p_count = (pred_uf.T == y_a).sum(axis=0)
  a_count = p_count
  count = 0
  # TODO : use complete inds to remove some from computation
  while((p_count > n_real*gamma*stol).sum() > 0) & (count < b_iter): # np.sum(pred == y_a) > n_real*gamma*stol): #n_real*tol/2):
    inds = p_count > n_real*gamma*stol
    print(F"Raising u_var for {inds.sum()} images with counts too high ")
    u_var[inds] = u_var[inds]*2
    #cov  = u_var*np.identity(n_sz)
    #samp = np.random.multivariate_normal(mean, cov, n_real)
    ################
    # already done aboveb_size = int((1024*2*2*2)/n_real)
    # want each batch to be about 4000, so n_real must be less than 4000ish
    # b_size = 
    # already done# pert_loader = torch.utils.data.DataLoader(dataset=images, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    u_var_s = u_var[inds]
    pert_loader = torch.utils.data.DataLoader(dataset=images[inds], batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))

    spred = torch.zeros([images[inds].shape[0], n_real])
    for j, data in enumerate(pert_loader, 0):
      gc.collect()
      n_sz = np.append(np.array(n_real), np.array(data.shape))
      mean = torch.zeros(tuple(n_sz))
      samp_t = torch.normal(mean, 1)
      # TODO : a_var subset must correspond with pre
      var_s = u_var_s[j*b_size:(j+1)*b_size]
      samp_t = samp_t*torch.sqrt(var_s)[:, None, None, None]
      #images_flat = images.flatten(start_dim = 1)
      #samp_t = Variable(torch.Tensor(samp))
      pert_a = data[None, :, :, :] + samp_t
      pert_af = pert_a.flatten(start_dim=0, end_dim=1)
      if torch.cuda.is_available():
        pert_af = pert_af.to("cuda")
  
      with torch.no_grad():
        ppred = torch.argmax(net(pert_af), axis=1)
      spred[j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)
    pred[inds] = spred #[j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)

    # ################
    # samp_t = torch.normal(mean, 1)
    # samp_t = samp_t*torch.sqrt(u_var)[:,None, None, None]
  
    # pert_a = images[None, :, :, :] + samp_t
    # pert_a_s = pert_a[:,inds]
    # if torch.cuda.is_available():
    #   pert_a_s = pert_a_s.to("cuda")
  
    # pert_af = pert_a_s.flatten(start_dim=0, end_dim=1)
    # b_size = 1024*2*2
    # pert_loader = torch.utils.data.DataLoader(dataset=pert_af, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    # pred = torch.zeros([pert_af.shape[0]])
    # for j, data in enumerate(pert_loader, 0):
    #   gc.collect()
    #   with torch.no_grad():
    #     ppred = torch.argmax(net(data), axis=1)
    #   pred[j*b_size:j*b_size+b_size] = ppred.detach()
  
    pred_uf = pred #[:,inds] = pred.reshape(tuple(pert_a_s.shape[0:2]))
    p_count = (pred_uf.T == y_a).sum(axis=0)  
    count += 1
    #print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
 
  # perform the bracketing 
  count = 0
  # TODO : use complete inds to eliminate some from computation
  for i in range(0,b_iter): #, a_var in enumerate(a_vars):
    count+=1
    inds = ((a_count >= torch.ceil(n_real*gamma*utol)) |
            (a_count < torch.floor(n_real*gamma*stol)))
    print(F"Persistence Bracketing : iteration {i+1}/{b_iter}, : Images within Tolerance: {len(inds)-inds.sum()}/{a_count.shape[0]} ")

    #rap[:,i] = a_count

    a_counts[i] = a_count
    a_vars[i] = a_var # save old variance

    #print("count:{}, interval: [{},{}]".format(a_counts[i], l_var, u_var))
		#floor and ceiling surround number
    # if everybody is within tolerance, then we're done, dump everything
    if (inds.sum() == 0):
        results["All_images_within_tolerance"] = True
        break
    u_inds = a_count > n_real*gamma*utol # counts too high, means variance too low
    l_inds = a_count < n_real*gamma*stol # counts too low, means variance too high
    # variance too high : this is now our upper bound 
    if (l_inds.sum() > 0):
      print(F"{l_inds.sum()} vars too high, lowering upper bounds")
      u_var[l_inds] = a_var[l_inds]
      a_var[l_inds] = (a_var[l_inds] + l_var[l_inds])/2
    # variance too low
    if (u_inds.sum() > 0):
      print(F"{u_inds.sum()} vars too low, raising lower bounds")
      l_var[u_inds] = a_var[u_inds]
      a_var[u_inds] = (u_var[u_inds] + a_var[u_inds])/2

    #print("Starting iteration: {}, sample variance: {}".format(count, a_var))
  		# compute sample and its torch tensor
    # TODO add in count of complete index to see how many we have left to find

    ################################################################
    # already done aboveb_size = int((1024*2*2*2)/n_real)
    # want each batch to be about 4000, so n_real must be less than 4000ish
    # b_size = 
    # already done# pert_loader = torch.utils.data.DataLoader(dataset=images, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    a_var_s = a_var[inds]
    pert_loader = torch.utils.data.DataLoader(dataset=images[inds], batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))

    #pred = torch.zeros([images.shape[0], n_real])
    spred = torch.zeros([images[inds].shape[0], n_real]) 
    for j, data in enumerate(pert_loader, 0):
      gc.collect()
      n_sz = np.append(np.array(n_real), np.array(data.shape))
      mean = torch.zeros(tuple(n_sz))
      samp_t = torch.normal(mean, 1)
      # TODO : a_var subset must correspond with pre
      var_s = a_var_s[j*b_size:(j+1)*b_size]
      samp_t = samp_t*torch.sqrt(var_s)[:, None, None, None]
      #images_flat = images.flatten(start_dim = 1)
      #samp_t = Variable(torch.Tensor(samp))
      pert_a = data[None, :, :, :] + samp_t
      pert_af = pert_a.flatten(start_dim=0, end_dim=1)
      if torch.cuda.is_available():
        pert_af = pert_af.to("cuda")
  
      with torch.no_grad():
        ppred = torch.argmax(net(pert_af), axis=1)
      spred[j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)
    pred[inds] = spred #[j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)

    #pred[inds][j*b_size:(j+1)*b_size] = ppred.detach().reshape(min(data.shape[0], b_size), n_real)
      #pred[j*b_size:(j+1)*b_size] = ppred.detach().reshape(b_size, n_real)

    ################################################################
    # samp_t = torch.normal(mean, 1)
    # samp_t = samp_t*torch.sqrt(a_var)[:,None, None, None]
  
    # pert_a = images[None, :, :, :] + samp_t
    # inds = ((a_count >= torch.ceil(torch.tensor(n_real*gamma*utol))) | (a_count < torch.floor(torch.tensor(n_real*gamma*stol))))
    # pert_a_s = pert_a[:,inds]
    # if torch.cuda.is_available():
    #   pert_a_s = pert_a_s.to("cuda")
  
    # pert_af = pert_a_s.flatten(start_dim=0, end_dim=1)
    # b_size = 1024*2*2
    # pert_loader = torch.utils.data.DataLoader(dataset=pert_af, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    # pred = torch.zeros([pert_af.shape[0]])
    # for j, data in enumerate(pert_loader, 0):
    #   gc.collect()
    #   with torch.no_grad():
    #     ppred = torch.argmax(net(data), axis=1)
    #   pred[j*b_size:j*b_size+b_size] = ppred.detach()
  
    pred_uf = pred #[:,inds] = pred.reshape(tuple(pert_a_s.shape[0:2]))

    # if torch.cuda.is_available():
    #   pert_a = pert_a.to("cuda")
  
    # pert_af = pert_a.flatten(start_dim=0, end_dim=1)
    # b_size = 1024*2*2
    # pert_loader = torch.utils.data.DataLoader(dataset=pert_af, batch_size = b_size, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    # pred = torch.zeros([pert_af.shape[0]])
    # for j, data in enumerate(pert_loader, 0):
    #   gc.collect()
    #   with torch.no_grad():
    #     ppred = torch.argmax(net(data), axis=1)
    #   pred[j*b_size:j*b_size+b_size] = ppred.detach()
  
    # pred_uf = pred.reshape(tuple(pert_a.shape[0:2]))
    a_count = (pred_uf.T == y_a).sum(axis=0)

  #return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}

  results["pred_heatmap_after"] = pred_uf
  results["vars"] = a_vars
  results["counts"] = a_counts
  results["var"] = a_var 
 #results["results_o"] = results_o # np.zeros([10,n_iter])
  return(results)

def orthant_bracketing (image, n_orth, gamma, n_real, n_iter, c_i, tol):
  if (gamma < (1-1/2**n_orth)):
    print(F"Gamma {gamma} too small for orthant dimension {n_orth} must be > { (1-1/2**n_orth)}")
    results = dict()
    results["a_sigs"] = HUGE#
    results["a_counts"] = n_real/2**n_orth
    results["a_frac"] = 1/2**n_orth
    results["a_var"] = 0.0#a_var
    #results["results_o"] = results_o # np.zeros([10,n_iter])
    return(results)

    return 
  utol = 1.0+tol
  stol = 1.0-tol
  # testing : image = -torch.ones([n_sz[1]])
  if (torch.linalg.norm(torch.abs(image)) < TEENY_WEENY):
    print("WARNING : image on orthant origin, returning 1/(2^dim)")
    results = dict()
    results["a_sigs"] = HUGE#
    results["a_counts"] = n_real/2**n_orth
    results["a_frac"] = 1/2**n_orth
    results["a_var"] = 0.0#a_var
    #results["results_o"] = results_o # np.zeros([10,n_iter])
    return(results)

  # start with point
  # generate n_real samples
  # count how many in orthant
  # grow/shrink
		# start with same magnitude noise as image
  a_var0 = 0.1 #np.var(image.detach().cpu().numpy())
  a_var = a_var0
  l_var = 0
  u_var = a_var*2
  a_vars = np.zeros(n_iter)
  # Adversarial image plus noise counts
  a_counts = np.zeros(n_iter)
  n_sz = np.array(image.unsqueeze(0).shape)
  n_sz[0] = n_real
  mean = np.zeros(n_sz)

  count = 0
  rap = np.zeros([n_real,n_iter])
		# grab the classification of the image under the network
  y_a = project_orthant(n_orth, image.unsqueeze(0))[1][0]
  samp = torch.tensor(np.random.normal(0,np.sqrt(u_var), n_sz)/np.sqrt(image.shape[0]))
  pert_a = image + samp #.data.cpu() + samp_t
  out_a = project_orthant(n_orth, pert_a)[1]

  #image_as = np.argmax(image_a,axis=1)
  #print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
  # plot counts for various gamma
  # # n_spread = 100
  # cspread = torch.linspace(0, u_var*100000, n_spread)
  # perts = torch.zeros([n_spread, 2000, 784])
  # counts = torch.zeros([n_spread])
  # for i in range(len(cspread)):
  #   perts[i] = image + torch.tensor(np.random.normal(0,np.sqrt(cspread[i]), n_sz)/np.sqrt(image.shape[0]))
  #   out_a = project_orthant(n_orth, perts[i])[1]
  #   counts[i] = (out_a == y_a).sum().float()
    
  # plt.plot(cspread, 2000-counts)
    
  #samp = samp
  ic = 0
  while (out_a == y_a).sum().float() > (n_real*gamma*stol):
    u_var = u_var*2
    #cov  = u_var*np.identity(n_sz)
    #samp = np.random.multivariate_normal(mean, cov, n_real)
    samp = torch.tensor(np.random.normal(0,np.sqrt(u_var), n_sz)/np.sqrt(image.shape[0]))
    pert_a = image + samp #.data.cpu() + samp_t
    out_a = project_orthant(n_orth, pert_a)[1]
    ic+=1
    if ic > 200:
      print(F"No resolution after {ic} iters : {(out_a == y_a).sum()}/{(n_real*gamma)}")
      break

    #print("u_var:{}, count:{}".format(u_var, np.sum(image_as == y_a)))
    
 
  # perform the bracketing 
  for i in range(0,n_iter): #, a_var in enumerate(a_vars):
    count+=1
    #print("Starting iteration: {}, sample variance: {}".format(count, a_var))
  		# compute sample and its torch tensor
    #cov  = u_var*np.identity(n_sz)
    #samp = np.random.multivariate_normal(mean, cov, n_real)
    samp = torch.tensor(np.random.normal(0,np.sqrt(a_var), n_sz)/np.sqrt(image.shape[0]))
    #samp = np.random.normal(0,np.sqrt(a_var),n_sz)
    #samp = np.random.normal(0,np.sqrt(a_var),m.shape)
    pert_a = image + samp #.data.cpu() + samp_t
    out_a = project_orthant(n_orth, pert_a)[1]

    rap[:,i] = out_a

    a_counts[i] = (out_a == y_a).sum()
    a_vars[i] = a_var # save old variance

    #print("count:{}, interval: [{},{}]".format(a_counts[i], l_var, u_var))
		#floor and ceiling surround number
    if ((a_counts[i] <= np.ceil(n_real*gamma*utol)) & (a_counts[i] >= np.floor(n_real*gamma*stol))):
        print(F"Found slam dunk variance {a_var} at {i+1}/{n_iter} with {a_counts[i]}/{n_real*gamma}")
        results = dict()
        results["a_sigs"] = a_vars
        results["a_counts"] = a_counts
        results["a_var"] = a_var
        #results["results_o"] = results_o # np.zeros([10,n_iter])
        return(results)
        #return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}        
    elif (a_counts[i] < n_real*gamma): # we're too high
        #print("We're too high, going from {} to {}".format(a_var, (a_var+l_var)/2)) 
        u_var = a_var
        a_var = (a_var + l_var)/2
    elif (a_counts[i] >= n_real*gamma): # we're too low
        l_var = a_var
        #print("We're too low,  going from {} to {}".format(a_var,(u_var+a_var)/2))
        a_var = (u_var + a_var)/2
        #u_var = u_var*2

  #return {"a_vars":a_vars, "a_counts":a_counts, "a_var":a_var}
  results = dict()
  results["a_sigs"] = a_vars
  results["a_counts"] = a_counts
  results["a_var"] = a_var
  #results["results_o"] = results_o # np.zeros([10,n_iter])
  return(results)

def o_sample(n_iin, n_cin, ni_iter, ni_real, vi_scal): #, oon_i):
  #ni_iter = 50
  #ni_real = 1000
  #vi_scal = 40

  results_o  = np.zeros([10,ni_iter])

  n_sz = n_iin.__len__()
  mean = np.zeros(n_sz)


  m_var = 0.04
  m_spacing = m_var*40/ni_iter
  a_sigs = np.linspace(0,m_var*vi_scal,ni_iter)
  count, e_count = 0, 0
  ssig = n_iter/5
  for a_sig in a_sigs:
      print("generating replication {}/{} with sigma {}".format(count,
                                                     ni_iter, a_sig))   
      cov  = a_sig*np.identity(n_sz)
      samp = np.random.multivariate_normal(mean, cov, ni_real)
      #m_samp = np.sqrt(np.var(samp,axis=1)/np.var(n_iin_a,axis=1))
      #figure(4)
      #plt.hist(m_samp)
      #plt.title("noise magnitude of the randomly added samples")
      # 5k samples
      i_samp_n = n_normalize(samp+n_iin) # normalize we'll try dividing by the max
      i_tens_n = Variable(torch.FloatTensor(i_samp_n))

      cpred_ao_n = np.argmax(nnet(i_tens_n).data.numpy(),axis=1)  

      a, b = np.unique(cpred_ao_n, return_counts=True)
      results_o[a,count] = b
      count += 1
  results = dict()
  results["a_sigs"] = a_sigs
  results["results_o"] = results_o # np.zeros([10,n_iter])
  return(results)

def n_sample(n_iin, ni_noises, ni_tar, ni_pred_a, ni_iter, ni_real, vi_scal, b_examples): #, oon_i):
  #ni_iter = 50
  #ni_real = 1000
  #vi_scal = 40

  n_sz = n_iin.__len__()
  mean = np.zeros(n_sz)

  if b_examples:
    results_eo = np.zeros([5, n_iin.shape[0]])
    results_eop = np.zeros([5])
    results_ea = np.zeros([len(ni_noises), 5, n_iin.shape[0]])
    results_eap = np.zeros([len(ni_noises), 5])

  results_a  = np.zeros([9,10,ni_iter])
  results_o  = np.zeros([10,ni_iter])
  results_c  = np.zeros([10,ni_iter])
  #results_oo = np.zeros([9,10,ni_iter])
  #oon_i = np.concatenate((oon_i, n_iin.reshape(1,784))).shape

  # m_var = np.var(ni_noises[1].detach().numpy())
  # we're going to keep this fixed for consistency
  m_var = 0.04
  m_spacing = m_var*40/ni_iter
  a_sigs = np.linspace(0,m_var*vi_scal,ni_iter)
  count, e_count = 0, 0
  ssig = n_iter/5
  for a_sig in a_sigs:
      print("generating replication {}/{} with sigma {}".format(count,
                                                     ni_iter, a_sig))   
      cov  = a_sig*np.identity(n_sz)
      samp = np.random.multivariate_normal(mean, cov, ni_real)
      #m_samp = np.sqrt(np.var(samp,axis=1)/np.var(n_iin_a,axis=1))
      #figure(4)
      #plt.hist(m_samp)
      #plt.title("noise magnitude of the randomly added samples")
      # 5k samples
      i_samp_n = n_normalize(samp+n_iin) # normalize we'll try dividing by the max
      i_tens_n = Variable(torch.FloatTensor(i_samp_n))

      cpred_ao_n = np.argmax(nnet(i_tens_n).data.numpy(),axis=1)  
      if b_examples & (count % ssig == 0):
        results_eo[e_count] = i_tens_n[0]
        results_eop[e_count] = cpred_ao_n[0]

      a, b = np.unique(cpred_ao_n, return_counts=True)
      results_o[a,count] = b

      i_samp_c = samp
      i_tens_c = Variable(torch.FloatTensor(i_samp_c))
      cpred_ao_c = np.argmax(nnet(i_tens_c).data.numpy(),axis=1)  
      a2, b2 = np.unique(cpred_ao_c, return_counts=True)
      results_c[a2,count] = b2
      for j in range(0, len(ni_noises)):
          n_iin_a = n_iin + ni_noises[j].detach().numpy()
          #o_iin_o = oon_i[j]
          #plt.imshow(n_iin_a.reshape(28,28))
          #plt.show()
          #print(n_iin_a.shape)
          i_samp_a = n_normalize(samp+n_iin_a)
          #o_iin_oo = n_normalize(o_iin_o)
          #print(samp.shape)
          #print(i_samp_a.shape)
          #plt.imshow(i_samp_a[0].reshape(28,28))
          #plt.show()
          i_tens_a = Variable(torch.FloatTensor(i_samp_a))
          #o_tens_oo = Variable(torch.FloatTensor(o_iin_oo))
          #plt.imshow(i_tens_a[0].reshape(28,28))
          #plt.show()
          cpred_ao = np.argmax(nnet(i_tens_a).data.numpy(),axis=1)
          #cpred_oo = np.argmax(nnet(o_tens_oo).data.numpy(),axis=1)          
          a, b = np.unique(cpred_ao, return_counts=True)
          results_a[j,a,count] = b
          #a, b = np.unique(cpred_oo, return_counts=True)
          #results_oo[j,a,count] = b
          if b_examples & (count % ssig == 0):
            results_ea[j, e_count] = i_tens_a[0]
            results_eap[j, e_count] = cpred_ao_n[0]
      if b_examples & (count % ssig == 0):            
        e_count+=1
      count += 1
  results = dict()
  results["a_sigs"] = a_sigs
  results["results_a"] = results_a # np.zeros([9,10,n_iter])
  results["results_o"] = results_o # np.zeros([10,n_iter])
  results["results_c"] = results_c # np.zeros([10,n_iter])
  #results["results_oo"] = results_oo # np.zeros([10,n_iter])  
  if (b_examples):
    results["results_eo"] = results_eo # np.zeros([10,n_iter])
    results["results_eop"] = results_eop # np.zeros([10,n_iter])
    results["results_ea"] = results_ea # np.zeros([10,n_iter])
    results["results_eap"] = results_eap # np.zeros([10,n_iter])
  return(results)


def n_samplot(results_eoi, cin, v_scal, a_sigs, results_oi, cpred_a, i_ind):
  i_disp = np.zeros([1*28,5*28])
  for i in range(0,5):
    i_disp[0*28:1*28, i*28:(i+1)*28] = results_eoi[i].reshape(28,28)
    #print(results_eoi[i])
  fig= plt.figure()
  plt.imshow(i_disp)
  fig.set_size_inches(20,4)
  fo = ddir+"/img/Image{}-O{}A{}_varx{}-example.png".format(i_ind, cin, cpred_a, v_scal)
  print("dumping to {}".format(fo))
  fig.savefig(fo, dpi=100)  
  #plt.show()
  
  plt.figure(num=1, figsize=(24,12), dpi=100)
  
  fig, ax= plt.subplots(2)#,sharex=True, sharey=True)
  fig.set_size_inches(24,12)
  
  roi = np.arange(0,10)
  if (cpred_a == "none"):
    ax[0].set_ylabel("Original")
    fos = np.setdiff1d(roi, np.array([cin]))
  else:
    ax[0].set_ylabel("Adversarial")   
    fos = np.setdiff1d(roi, np.array([cin, cpred_a]))
    ax[0].plot(a_sigs,results_oi[cpred_a].T, linewidth=2.0, color="red")    

  print("shape of results_oi: {}".format(results_oi.shape))
  ax[0].plot(a_sigs,results_oi[fos].T, linewidth=2.0, color="blue")
  ax[0].plot(a_sigs,results_oi[cin].T, linewidth=2.0, color="black")

  if (cpred_a == "none"):
    ax[0].legend(np.concatenate([fos, np.array([cin])]),loc='upper right')
  else:
    ax[0].legend(np.concatenate([np.array([cpred_a]),
                                 fos,
                                 np.array([cin])]),loc='upper right')
  ax[1].imshow(i_disp)#(a_sigs,results_a.T, linewidth=2.0)
  
  ax[1].plot(a_sigs,results_a.T, linewidth=2.0)
  ax[1].set_ylabel("adversarial")
  ax[1].legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
  

  
  ax[2].plot(a_sigs,results_c.T, linewidth=2.0)
  ax[2].set_ylabel("noise")
  ax[2].legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
  ax.subplot(2,2,4)
  ax.plot(a_sigs,results_z.T, linewidth=2.0)
  ax.title("zeros")
  ax.legend([0,1,2,3,4,5,6,7,8,9],loc='upper right')
  fo = ddir+"/img/Image{}-O{}A{}_varx{}.png".format(i_ind, cin, cpred_a, v_scal)
  print("generating: {}".format(fo))
  fig.subplots_adjust(hspace=0)
  fig.set_size_inches(24,12)
  fig.savefig(fo, dpi=100)


class Normalize(nn.Module) :
  def __init__(self, mean, std) :
      super(Normalize, self).__init__()
      self.register_buffer('mean', torch.Tensor(mean))
      self.register_buffer('std', torch.Tensor(std))
  def forward(self, input):
      # Broadcasting
      mean = self.mean.reshape(1, 3, 1, 1)
      std = self.std.reshape(1, 3, 1, 1)
      #input = transforms.Resize(256)(input)
      #input = transforms.CenterCrop(224)(input)
      return (input - mean) / std

def write_header(sched, fo, cpus, mem, gpus, group, wtime, ctime):
    if sched.lower() == "pbs":
      header = ("#!/bin/bash" + "\n" + 
      "# Your job will use 1 node, {} cores, and {}gb of memory total.".format(cpus, mem) +
      "\n" + 
      "#PBS -q windfall" + "\n" + 
      " " + "\n" + 
      "#PBS -l select=1:ncpus={}:mem={}gb:np100s=1:os7=True".format(cpus, mem) + "\n" + 
      "### Specify a name for the job" + "\n" + 
      "#PBS -N {}".format(fo) + "\n" + 
      "### Specify the group name" + "\n" + 
      "#PBS -W group_list=rwg2" + "\n" + 
      "### Used if job requires partial node only" + "\n" + 
      "###PBS -l place=pack:shared" + "\n" + 
      "### CPUtime required in hhh:mm:ss." + "\n" + 
      "### Leading 0's can be omitted e.g 48:0:0 sets 48 hours" + "\n" + 
      "###PBS -l cputime=120:20:00" + "\n" + 
      "### Walltime is how long your job will run" + "\n" + 
      "#PBS -l walltime={}".format(wtime) + "\n" + 
      "### Joins standard error and standard out" + "\n" + 
      "###PBS -k oed" + "\n" +
      "#PBS -e err/{}-error.txt\n".format(fo) + 
      "#PBS -o err/{}-output.txt\n".format(fo) + 
      "cd code/stab" + "\n" + 
      "module load pytorch" + "\n")
    elif sched.lower() == "slurm":
      header = ("#!/bin/bash" + "\n" + 
      "#SBATCH --job-name={}\n".format(fo) +  # ignoring mem for now
      "#SBATCH --time {}\n".format(wtime) +  # ignoring mem for now                
      "#SBATCH -e err/{}-error.txt\n".format(fo) +
      "#SBATCH -o err/{}-error.txt\n".format(fo) +
      "#SBATCH --account=rwg2\n" +
      "#SBATCH --partition=standard\n" +    # replace with windfall?
      "#SBATCH --nodes=1\n" +
      "#SBATCH --ntasks={}\n".format(cpus) +
      "#SBATCH --mem={}gb\n".format(mem) +  
      #"#SBATCH --mem-per-cpu=10gb\n" +  
      "#SBATCH --gres=gpu:{}\n".format(gpus))  
      # could build the job handler into python -- put marker files to prevent stepping on other jobs...
      # could chop them up into reasonable chunks (inside job_writer) of a reasonable workload size so things won't overlap anyway
      # ideal : have N workers -- send my jobs in sequence to the workers so that each worker has K jobs running at once -- if things crash, make it re-entrant for mid-failed jobs and let it know some jobs have completed and don't need to be run again. 
      # need to know: Optimal k = jobs/node
      # need to mark jobs complete somehow (marker files?)
                
    return header
                 
def project_orthant(odim, points, thetas):
  # points should be a torch tensor
  # are we in or out?
  #tests = torch.tensor([[0.5,1,3,-1,-2,21,1,-1,2,5],
  #                      [0.5,1,3,11,2,21,1,1,2,5],
  #                      [0.5,1,3,11,2,0,1,1,2,5]])
  #points = torch.zeros([3,test.shape[0]])
  #points = tests
  if type(thetas) is int:
    thetas = np.ones(odim-1)*thetas
  elif type(thetas) is np.ndarray:
    if (len(thetas) < (odim-1)):
      print(F"Warning not enough thetas {len(thetas)} for odim {odim}")
  elif type(thetas) is torch.Tensor:
    if (len(thetas) < (odim-1)):
      print(F"Warning not enough thetas {len(thetas)} for odim {odim}")
  else:
    print(F"Warning unknown thetas type: {type(thetas)}")
    
  points = torch.randn([10000,100])/10
  points_out = points.clone().detach()
  # comparison matrix
  thetas_deg = torch.tensor((thetas*np.pi)/180)
  comp = points[:,0:odim].clone().detach()
  for i in range(len(thetas)):
    comp[:,i+1] = comp[:,0]*torch.tan(thetas_deg[i])
  # just check and project onto x_0
  comp[:,0] = HUGE
  for i in range(odim-1):
    
  # set all the coordinates outside the orthant to their comp
  # note this projection is referenced to x_0
  points_out[:,0:odim][(points_out[:,0:odim] < comp) & (points_out[:,0:odim] > 0.0)] = comp
  
  # note argmax is arbitrarily choosing one dimension to project onto
  points_out[:,0:odim][tuple(np.arange(0,points_out.shape[0])),
                   tuple(points_out[:,0:odim].argmin(dim=1))] = 0.0
  # 0 for outside, 1 for inside
 
  ssampt = points[:,0:odim]
  element_countt = ((ssampt[:,1:odim] < comp) & (ssampt > 0.0)).sum(axis=1)
  class_out = (element_countt >= odim).int()
  #orthant_countt = (element_countt >= dim).sum()
  #fract = orthant_countt/n_samp
  
  return(points_out, class_out)

def db_bracket (net, oimg, aimg, oclass=None, aclass=None, 
             lb=0.0, ub=1.0, guess=0.8, tol=torch.finfo(float).eps, niter=100):
  lstep, ustep, step = torch.tensor(lb).double(), torch.tensor(ub).double(), torch.tensor(guess).double()
  if oclass == None:
    oclass = torch.argmax(net(oimg))
  if aclass == None:
    aclass = torch.argmax(net(aimg))
  nstep = niter
  steps = torch.zeros(nstep)
  classes = torch.zeros(nstep).int()
  diff = aimg - oimg
  # oclass, aclass = 2, 0
  ostep = torch.tensor(0.0).double()
  astep = torch.tensor(1.0).double()
  for j in range(nstep):
    img_i = oimg + step*(diff)
    with torch.no_grad():
      if torch.cuda.is_available():
        img_i = img_i.to('cuda')
      out = net(img_i)
    oopred = int(torch.argmax(out))
    steps[j] = step
    classes[j] = oopred
    #print(F"lstep: {lstep}, ustep: {ustep}, step: {step} -> class: {oopred}")
    if torch.abs(ostep - astep) < tol:
      #print(F"ostep - astep = {ostep - astep} < {tol} (tol), DONE!")
      break
    if oopred == oclass:
      #print(F"lstep {lstep} -> {step}")
      lstep = step
      if step > ostep:
        ostep = step
    elif oopred == aclass:
      #print(F"ustep {ustep} -> {step}")
      ustep = step
      if step < astep:
        astep = step
  
    step = (lstep + ustep)/2
  odict = {}
  odict["ostep"] = ostep
  odict["astep"] = astep
  odict["lstep"] = lstep
  odict["ustep"] = ustep
  odict["step"] = step
  odict["img"] = img_i 
  odict["aimg"] = oimg + astep*(diff)
  odict["oimg"] = oimg + ostep*(diff)
  odict["n_steps"] = j
  return odict             

def db_bracket_tensor (net, oimg, aimg, oclass=None, aclass=None, 
             lb=0.0, ub=1.0, guess=0.8, tol=torch.finfo(float).eps, niter=100):
  lstep, ustep, step = torch.ones(oimg.shape[0])*lb, torch.ones(oimg.shape[0])*ub, torch.ones(oimg.shape[0])*guess
  if oclass == None:
    oimg = oimg.reshape(oimg.shape)
    oclass = torch.argmax(net(oimg), axis=1)
  if aclass == None:
    aimg = aimg.reshape(oimg.shape)
    aclass = torch.argmax(net(aimg), axis=1)
  nstep = niter
  steps = torch.zeros((nstep, oimg.shape[0]))
  classes = torch.zeros((nstep, oimg.shape[0])).int()
  diff = aimg - oimg
  # oclass, aclass = 2, 0
  ostep = torch.zeros(oimg.shape[0]) #tensor(0.0).double()
  astep = torch.ones(aimg.shape[0]) #astep = torch.tensor(1.0).double()
  for j in range(nstep):
    img_i = oimg + step[:,None,None,None]*(diff)
    oopred = torch.argmax(net(img_i), axis=1)#pred.reshape(tuple(img_i.shape[0:2]))

    # with torch.no_grad():
    #   if torch.cuda.is_available():
    #     img_i = img_i.to('cuda')
    #   out = net(img_i)
    # oopred = int(torch.argmax(out))
    steps[j] = step
    classes[j] = oopred
    #print(F"lstep: {lstep}, ustep: {ustep}, step: {step} -> class: {oopred}")
    if (torch.abs(ostep - astep) < tol).sum() == len(ostep):
      #print(F"ostep - astep = {ostep - astep} < {tol} (tol), DONE!")
      break
    # largest step step on the original side
    linds = (oopred == oclass)
    lstep[linds] = step[linds]
    # keep the largest step on this side (separate for some reason?)
    oinds = (step > ostep) & linds
    ostep[oinds] = step[oinds]

    uinds = (oopred == aclass)
    ustep[uinds] = step[uinds]
    ainds = (step < astep) & uinds
    astep[ainds] = step[ainds]
    # count number of lsteps in oclass and usteps in aclass, make sure always 100,100
    img_l = oimg + lstep[:,None,None,None]*(diff)
    img_u = oimg + ustep[:,None,None,None]*(diff)
    lpred = torch.argmax(net(img_l), axis=1)#pred.reshape(tuple(img_i.shape[0:2]))
    upred = torch.argmax(net(img_u), axis=1)#pred.reshape(tuple(img_i.shape[0:2]))

    
    print(F"number in oclass: {linds.sum()}, aclass: {uinds.sum()}")
    print(F"Mean Gap vs tol: {torch.abs(ustep - lstep).mean()} vs {tol}")
    # if oopred == oclass:
    #   #print(F"lstep {lstep} -> {step}")
    #   lstep = step
    #   if step > ostep:
    #     ostep = step
    # elif oopred == aclass:
    #   #print(F"ustep {ustep} -> {step}")
    #   ustep = step
    #   if step < astep:
    #     astep = step
  
    step = (lstep + ustep)/2
  odict = {}
  odict["ostep"] = ostep
  odict["astep"] = astep
  odict["lstep"] = lstep
  odict["ustep"] = ustep
  odict["step"] = step
  odict["img"] = img_i 
  odict["uimg"] = oimg + ustep[:,None,None,None]*(diff)
  odict["limg"] = oimg + lstep[:,None,None,None]*(diff)
  odict["n_steps"] = j
  return odict #img_i, step

def interp(net, img1, img2, n_interp):
  pspace = torch.linspace(0, 1, n_interp)
  pshape = np.array(img1.shape)
  pshape[0] = n_interp
  imgs = torch.ones(tuple(pshape))
  imgs = imgs*pspace[:,None,None,None]#torch.matmul(imgs, pspace)

  #pspace = torch.linspace(0, 1, n_interp)
  #pshape = np.array(c1_img.shape)
  #pshape[0] = n_interp
  #imgs = torch.ones(tuple(pshape))
  #imgs = imgs*pspace[:,None,None]#torch.matmul(imgs, pspace)
  #img1_t = t.forward(img1[None]) # size: [1,k]
  #img2_t = t.forward(img2[None]) # size: [1,k]
  #imgs_t = torch.ones(tuple(img2_t)) # size: [1,k]
  #imgs_t = imgs*pspace[:,None,None,None] # size: [n_interp, k]
  #pimgs_t = (img1_t[0][None,:,:,:] + imgs_t*(img2_t[0][None,:,:,:] - img1_t[0][None,:,:,:])) # size: [n_interp, k]
  #pimgs = t.backward(pimgs_t) # size: [n_interp, 1, 28, 28]
  
  pimgs = (img1[0][None,:,:,:] + imgs*(img2[0][None,:,:,:] - img1[0][None,:,:,:]))
  pimgs = pimgs.reshape(pimgs.shape)
  pclas = torch.argmax(net(pimgs), axis=1)
  return pimgs, pclas
  
def boundary_sample (net, img, var, lclass, rclass, n_samp = 100, ssize=0.01):
  n_iter = 30
  # TODO :
  #   1. take gaussian
  mshape = torch.tensor(img.shape)
  mshape[0] = n_samp
  mean = torch.zeros(tuple(mshape.int()))
  samp_t = torch.normal(mean, torch.ones(img.shape[0])*var)
  #samp_t = samp_t*torch.sqrt(a_var)[:,None, None, None]

  asample = (img+samp_t).clone()
  osample = asample.clone()
  oclass = torch.argmax(net(imgs), axis=1)
  aclass = oclass.clone()
  tclass = oclass.clone()
  tclass[oclass == lclass] = rclass
  tclass[oclass == rclass] = lclass
  #   2. compute gradient to other class
  # DONE : check that this is tensorized -- appears to be
  output_fil = nn.CrossEntropyLoss()  # error function
  ograd = attack_grad(net, asample, tclass, output_fil)
  #   3. loop take steps for each image that is not yet in the other class
  count = 0

  while ((aclass == oclass).sum() > 0):
    #     i. (optional) update gradient after each step
    ograd = attack_grad(net, asample, tclass, output_fil)
    asample = asample-ograd*(1+ssize)
    aclass = torch.argmax(net(asample), axis=1)
    print(F"Iter: {count} : Number of samples at original class: {(aclass == oclass).sum()}/{len(oclass)}")
    print(F"Iter: {count} : Number of samples at target class: {(aclass == tclass).sum()}/{len(oclass)}")

    if (count > n_iter):
      print(F"Hit Iteration Limit: {count+1}/{n_iter}")
      break
    count+=1

  #   4. tensorized bracketing to compute points between here and the attack.
  # TODO : confirm that these are coming back correctly
  #      get u_step and l_step and make sure we're actually at the db
  db_report = db_bracket_tensor(net, osample, asample)
  db_imgs = db_report["img"] 
  db_oclass = torch.argmax(net(db_report["oimg"]), axis=1)
  db_aclass = torch.argmax(net(db_report["aimg"]), axis=1)
  # want images,
  # size of image-step (distance to decision boundary)
  db_dist = img_db - osample
  
  oclass = torch.argmax(net(db_imgs), axis=1)
  tclass = oclass.clone()
  tclass[oclass == lclass] = rclass
  tclass[oclass == rclass] = lclass

  db_grad = attack_grad(net, db_imgs, tclass, output_fil)
  # gradients of images near the decision boundary
  ggrads   = ograd# gradients of the gaussians going to the opposite class
  # for each of these, do bracketing. 
  
  # make a bunch of adversaries
  oimgs = oimg_i.detach().clone()
  oouts = net(oimgs)
  oclasss = torch.argmax(oouts, axis=1)
  n_steps = 50
  epsilon = 0.5
  aimgs = (oimg_i - epsilon*ggrads).detach().clone()
  # skip all of this and just make the opposite point the adversary. 
  for i in range(n_steps):
    ograd = attack_grad(aimgs, gclasses, ain, gtarg, net, niter,
                          output_fil, opmeth)
    aimgs = aimgs - epsilon*ggrads
    aouts = net(aimgs)
    aclasss = torch.argmax(aouts, axis=1)
    print(F"Iter : ({i}) - (aclasss == oclasss).sum() = {(aclasss == oclasss).sum()}")
    print(F"Iter : ({i}) - (aclasss == gtarg).sum() = {(aclasss == gtarg).sum()}")
    if ((aclasss == gtarg).sum() == len(aclasss)) & ((aclasss == oclasss).sum() == 0):
      break
  
  # if ((aclasss == gtarg).sum() == len(aclasss)) & ((aclasss == oclasss).sum() == 0):
  #   print("Attack Success")
  # else:
  #   print("Attack Failed for uvar = {uvar} oclass, gtarg : {(aclasss == oclasss).sum()}, {(aclasss == gtarg).sum()}")
  #   continue
    
  gbounds  = torch.zeros(oimgs.shape)# iteratively produced points on the decision boundary
  for i in range(len(oimgs)):
    oimg, oclass = oimgs[i].unsqueeze(0), int(oclasss[i])
    aimg, aclass = aimgs[i].unsqueeze(0), int(aclasss[i])  
    print(F"Bracketing image {i+1}/{len(oimgs)}")
    odict = bracket(oimg, oclass, aimg, aclass, net)
    diff2 = aimg - oimg
    astep = odict["astep"]
    gbounds[i] = (oimg + astep*(diff2))[0]


