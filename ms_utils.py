import casatools
import casatasks
import numpy as np
import glob
import os


def prepare_ms_file(ms_file):
    """
    This function will:
      1. bin the data to have only 1 channel per spw (we assume continuum only)
      2. split the ms_file in 1 file per execution
      3. export each individual execution in a npz file
    """

    # Averaging over spectral window and removing flags
    split_name = os.path.splitext(ms_file)
    filename = split_name[0] ; ext = split_name[1]
    ms_file_avg = filename+"_avg"+ext
    os.system("rm -rf " + filename+"_avg*")  # removing existing structure
    casatasks.mstransform(vis=ms_file, outputvis=ms_file_avg, datacolumn='data', chanaverage=True, chanbin=3840, keepflags=False)

    # splitting the main ms file into individual observations
    split_name =  filename+"_avg_split_"
    split_all_obs(ms_file_avg, split_name)

    # finding appropriate files
    ms_files = glob.glob(split_name + "*.ms")

    # exporting each split ms into a npz file
    for i, file in enumerate(ms_files):
        export_MS(file)


# The functions below are adapted from the DSHAP reduction utils and updated to work with modular casa
def split_all_obs(msfile, nametemplate):
    """

    Split out individual observations in a measurement set

    Parameters
    ==========
    msfile: Name of measurement set, ending in '.ms' (string)
    nametemplate: Template name of output measurement sets for individual observations (string)
    """
    tb = casatools.table()
    tb.open(msfile)
    spw_col = tb.getcol('DATA_DESC_ID')
    obs_col = tb.getcol('OBSERVATION_ID')
    field_col = tb.getcol('FIELD_ID')
    tb.close()

    obs_ids = np.unique(obs_col)
    #yes, it would be more logical to split out by observation id, but splitting out by observation id in practice leads to some issues with the metadata
    for i in obs_ids:
        spws = np.unique(spw_col[np.where(obs_col==i)])
        fields = np.unique(field_col[np.where(obs_col==i)]) #sometimes the MS secretly has multiple field IDs lurking even if listobs only shows one field
        if len(spws)==1:
            spw = str(spws[0])
        else:
            spw = "%d~%d" % (spws[0], spws[-1])

        if len(fields)==1:
            field = str(fields[0])
        else:
            field = "%d~%d" % (fields[0], fields[-1])

        #start of CASA commands
        outputvis = nametemplate+'%d.ms' % i
        os.system('rm -rf '+outputvis)
        print("#Saving observation %d of %s to %s" % (i, msfile, outputvis))
        casatasks.split(vis=msfile,
              spw = spw,
              field = field,
              outputvis = outputvis,
              datacolumn='data')


def export_MS(msfile):
    """
    Spectrally averages visibilities to a single channel per SPW and exports to .npz file

    msfile: Name of CASA measurement set, ending in '.ms' (string)
    """

    print("#-- Processing to "+msfile)

    filename = msfile
    if filename[-3:]!='.ms':
        print("MS name must end in '.ms'")
        return
    # strip off the '.ms'
    MS_filename = filename.replace('.ms', '')


    # get information about spectral windows
    tb = casatools.table()
    tb.open(MS_filename+'.ms/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()

    # spectral averaging (1 channel per SPW)
    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    casatasks.split(vis=MS_filename+'.ms', width=num_chan, datacolumn='data',
    outputvis=MS_filename+'_spavg.ms')

    # get the data tables
    #tb.open(MS_filename+'_spavg.ms')
    tb.open(MS_filename+'.ms')
    data   = np.squeeze(tb.getcol("DATA"))
    flag   = np.squeeze(tb.getcol("FLAG"))
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    ant1     = tb.getcol("ANTENNA1")
    ant2     = tb.getcol("ANTENNA2")
    time   = tb.getcol("TIME")
    tb.close()


    print(" Testing flags ....")
    for k, f in enumerate(flag[0,:]):
        if f:
            print(k, f, flag[0,k], flag[1,k])
    for k, f in enumerate(flag[1,:]):
        if f:
            print(k, f, flag[0,k], flag[1,k])
    print("Done")

    print(" Testing time ....")
    print(is_sorted(time))
    print(" Done")

    # get frequency information
    tb.open(MS_filename+'_spavg.ms/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    if (len(freqlist.shape) == 0):
        freqlist = np.array([freqlist])
    tb.close()

    # get rid of any flagged columns
    good   = np.squeeze(np.any(flag, axis=0)==False)

    print("Selecting", len(np.where(good)[0]), "/", flag.shape, time.shape, ant1.shape, ant2.shape)
    #ValueError




    data   = data[:,good]
    flag = flag[:,good]
    weight = weight[:,good]
    uvw    = uvw[:,good]
    spwid = spwid[good]

    ant1 = ant1[good]
    ant2 = ant2[good]
    time = time[good]

    # compute spatial frequencies in lambda units
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid) #get spectral frequency corresponding to each datapoint
    u = uvw[0,:] * freqs / 2.9979e8
    v = uvw[1,:] * freqs / 2.9979e8

    #average the polarizations
    Wgt = np.sum(weight, axis=0)
    Re  = np.sum(data.real*weight, axis=0) / Wgt
    Im  = np.sum(data.imag*weight, axis=0) / Wgt

    for k, r in enumerate(Re):
        if not np.isfinite(r):
            print(k, r)
            print(data.shape, flag.shape)
            print(data[:,k].real)
            print(weight[:,k])
            print(flag[:,k])
            print("----")
            #raise ValueError("Non-finite values found in Vis")

    Vis = Re + 1j*Im


    #output to npz file and delete intermediate measurement set
    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    os.system('rm -rf '+MS_filename+'.vis.npz')
    np.savez(MS_filename+'.vis', u=u, v=v, Vis=Vis, Wgt=Wgt, ant1=ant1, ant2=ant2, time=time, spwid=spiwd)
    print("#Measurement set exported to %s" % (MS_filename+'.vis.npz',))
