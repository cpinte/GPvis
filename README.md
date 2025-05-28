python file:
- uvdenoise.py was my initial code
- uvdenoise_eilish.py seems to only change a few weights in the kernel
- refine_visibilities.py : add sort on y and yerr + iterative scheme to filter out errors, might just need a with when opening files (see uvdenoise.py)

casa files:
- export_ms.py : CASA, make npz files
- update_ms.py : CASA, copy ms files and replace values


- We will need to split or merge the spws too, as they appear one after the other in the npz file now
(need to remake npz files)


Todo:
 - split or extract spw to avoid confusion
 - merge casa and python code
 - try with just average to make it to the end (ie make a dirty image)
 - make parallel
