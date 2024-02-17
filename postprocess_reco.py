import pandas as pd
import sys
import numpy as np
import uproot
import matplotlib.pyplot as plt
from time import time
#My imports
sys.path.append('/exp/sbnd/app/users/brindenc/mysbnana_v09_75_03/srcs/sbnana/sbnana/SBNAna/pyana')
s0 = time()
from sbnd.cafclasses.pfp import PFP
from sbnd.cafclasses.slice import CAFSlice
from sbnd.constants import *
from sbnd.numu.numu_constants import *
from sbnd.prism import PRISM_BINS
s1 = time()

print(f'My imports: {s1-s0:.2f} s')

#Constants/variables
DATA_DIR  = '/exp/sbnd/data/users/brindenc/analyze_sbnd/numu/v09_78_04_wc_pandora'
#DATA_DIR = '/exp/sbnd/data/users/brindenc/ML/test_fcl/debug_trackid/v4'
#FNAME = 'single.df'
FNAME = 'all.df'
#FNAME = 'test.df'
NOM_POT = 0.6e20 # stats for first run
ISMC = True #is this MC or data
APPLY_CUTS = False #apply cuts to the data

pfp = PFP(pd.read_hdf(f'{DATA_DIR}/{FNAME}', key='pfp')
          ,pot=NOM_POT
          ,prism_bins=PRISM_BINS
          ,momentum_bins=MOMENTUM_BINS
          ,costheta_bins=COSTHETA_BINS)

slc = CAFSlice(pd.read_hdf(f'{DATA_DIR}/{FNAME}', key='slice')
               ,pot=NOM_POT
               ,prism_bins=PRISM_BINS)
s2 = time()
print(f'Load df time: {s2-s1:.2f} s')

#cut these by default since there was no reco attempt
print('-cutting nu_score < 0')
slc.cut_has_nuscore(cut=True) 
pfp.data = slc.get_reference_df(pfp) #cut pfp to only those in slice

#PFP processing
pfp.fix_shw_energy(fill=np.nan,dummy=np.nan)
pfp.add_pfp_semantics() #do this before cleaning
#pfp.clean(dummy_vals=[-9999,-999,999,9999,-5])
pfp.add_reco_containment()
pfp.add_neutrino_dir()
pfp.add_theta()
pfp.add_bestpdg(method='x2')
pfp.add_trk_bestenergy()
pfp.add_Etheta()
pfp.add_stats()
s3 = time()
print(f'pfp time: {s3-s2:.2f} s')

#Slice processing
#slc.clean(dummy_vals=[-9999,-999,999,9999,-5])
slc.add_has_trk(pfp)
slc.add_has_muon(pfp)
slc.add_shws_trks(pfp)
slc.add_in_av()
slc.add_event_type()
slc.add_tot_visE()
#Cuts
slc.cut_cosmic(cut=APPLY_CUTS,fmatch_score=7,nu_score=0.4)
slc.cut_fv(cut=APPLY_CUTS)
slc.cut_trk(cut=APPLY_CUTS)
slc.cut_muon(cut=APPLY_CUTS)
if not APPLY_CUTS: slc.cut_all() #display if sample survives all cuts
s4 = time()
print(f'slice time: {s4-s3:.2f} s')

#Final cleanup
pfp.data.sort_index(inplace=True)
slc.data.sort_index(inplace=True)
s5 = time()
print(f'cleanup time: {s5-s4:.2f} s')

#Save to hdf5
save_name = f'{DATA_DIR}/{FNAME.split(".")[0]}_processed.df'
pfp.data.to_hdf(save_name, key='pfp')
slc.data.to_hdf(save_name, key='slice')
s6 = time()
print(f'saved to {save_name} : {s6-s5:.2f} s')



