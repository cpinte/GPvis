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

    if ms_file[-3:]!='.ms':
        print("MS name must end in '.ms'")
        return

    filename = ms_file.replace(".ms","")

    # Removing existing structure
    clean_spavg()

    # Flagging any potential NaN
    ms_file_no_NaN = filename+"_no_NaN.ms"
    flag_NaN(ms_file, ms_file_no_NaN)

    # Averaging over spectral window and removing flags
    print(" ")
    ms_file_avg = filename+"_spavg.ms"
    spectral_avg(ms_file_no_NaN, ms_file_avg)

    # splitting the main ms file into individual observations
    print(" ")
    split_name =  filename+"_spavg_split_"
    split_all_obs(ms_file_avg, split_name)

    # exporting each split ms into a npz file
    ms_files = glob.glob(split_name + "*.ms")
    for i, file in enumerate(ms_files):
        print(" ")
        export_MS(file)


def clean_spavg():
    """
    Remove all spectrally averaged files
    """
    os.system("rm -rf *spavg*")


# The functions below are adapted from the DSHAP reduction utils and updated to work with modular casa
def spectral_avg(ms_file,outputvis):
    """
    Average all spectral windows to 1 channel
    """

    print("# Spectrally averaging and removing flagged data from %s to %s" % (ms_file, outputvis))

    # get information about spectral windows
    tb = casatools.table()
    tb.open(ms_file+'/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()

    # spectral averaging (1 channel per SPW)
    os.system('rm -rf %s' % outputvis)
    casatasks.split(vis=ms_file, width=num_chan, datacolumn='data',outputvis=outputvis,keepflags=False)


def split_all_obs(ms_file, nametemplate):
    """
    Split out individual observations in a measurement set

    Parameters
    ==========
    ms_file: Name of measurement set, ending in '.ms' (string)
    nametemplate: Template name of output measurement sets for individual observations (string)
    """

    print("# Splitting %s" % ms_file)

    tb = casatools.table()
    tb.open(ms_file)
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
        print("   - Saving observation %d of %s to %s" % (i, ms_file, outputvis))
        casatasks.split(vis=ms_file,
              spw = spw,
              field = field,
              outputvis = outputvis,
              datacolumn='data')


def export_MS(ms_file):
    """
    Export file to .npz file

    ms_file: Name of CASA measurement set, ending in '.ms' (string)
    """

    print("# Exporting "+ms_file)

    # get the data tables
    tb = casatools.table()
    tb.open(ms_file)
    data   = np.squeeze(tb.getcol("DATA"))  # Note the squeeze here
    flag   = np.squeeze(tb.getcol("FLAG"))
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    ant1     = tb.getcol("ANTENNA1")
    ant2     = tb.getcol("ANTENNA2")
    time   = tb.getcol("TIME")
    tb.close()

    if np.any(flag):
        print("  ---- WARNING --- : data has flags")
    else:
        print("  - No flags: OK")

    if np.all(np.isfinite(data.real)):
        print("  - real part: OK")
    else:
        print("  ---- WARNING --- : real part has NaN")
        print(np.all(np.isfinite(data.real)), np.any(np.isnan(data.real)), np.all(np.isfinite(data)), np.any(np.isnan(data)))

    if np.all(np.isfinite(data.imag)):
        print("  - imaginary part: OK")
    else:
        print("  ---- WARNING --- : imaginary part has NaN")
        print(np.all(np.isfinite(data.imag)), np.any(np.isnan(data.imag)), np.all(np.isfinite(data)), np.any(np.isnan(data)))


    # get frequency information
    tb.open(ms_file+'/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    if (len(freqlist.shape) == 0):
        freqlist = np.array([freqlist])
    tb.close()

    # compute spatial frequencies in lambda units
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid) #get spectral frequency corresponding to each datapoint
    u = uvw[0,:] * freqs / 2.9979e8
    v = uvw[1,:] * freqs / 2.9979e8

    #average the polarizations
    Wgt = np.sum(weight, axis=0)
    Re  = np.sum(data.real*weight, axis=0) / Wgt
    Im  = np.sum(data.imag*weight, axis=0) / Wgt

    Vis = Re + 1j*Im

    npz_file = ms_file.replace('.ms', '.viz.npz')
    os.system('rm -rf '+npz_file)
    np.savez(npz_file, u=u, v=v, Vis=Vis, Wgt=Wgt, ant1=ant1, ant2=ant2, time=time, spwid=spwid)
    print(" --->  Measurement set exported to %s" % (npz_file))


def flag_NaN(ms_file,outputvis):

    print("# Flagging NaN from %s to %s" % (ms_file, outputvis))

    # copy original ms file into new file
    os.system("rm -rf " + outputvis)
    os.system("cp -r " + ms_file + " " + outputvis)

    tb = casatools.table()
    tb.open(outputvis, nomodify=False)
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")

    # Adding a flag on data with NaN on real or imaginary part
    #isnan = np.isnan(data)
    isnan = np.logical_not(np.isfinite(data))

    if (np.any(isnan)):
        print("   - Data table has NaN, flagging them")
        flag = flag | isnan

        # Updating flags in ms file
        tb.putcol("FLAG",flag)
        tb.flush()
    else:
        print("   - No NaN in "+ms_file)

    tb.close()





def update_ms(target):
    """ FUNCTION THAT IS CALLED FROM MAIN.py """
    #target = "Elias24"

    # finding appropriate ms files
    search_name = target + "_cont_avg_split_*.ms"
    ms_files = glob.glob(search_name)
    model_names = []

    # iterating over found files and updating visibility
    for i, ms_file in enumerate(ms_files):
        # create model name
        model = ms_file.replace('cont_avg_split', 'model')      # output ms
        model_names.append(model)

        # find npz file
        npz_file = ms_file.replace('.ms', '_updated.vis.npz')

        # load visibility data from npz file
        vis = dict(np.load(npz_file))['Vis']

        # copy original ms file into new file
        os.system("rm -rf " + model)
        os.system("cp -r " + ms_file + " " + model)

        # open model dataset
        tb.open(model)
        data = tb.getcol("DATA")    # grab existing structure
        flag = tb.getcol("FLAG")
        tb.close()

        # note flagged columns
        flagged = np.all(flag, axis=(0, 1))
        unflagged = np.squeeze(np.where(flagged == False))

        # replace original data with updated vis
        data[:, :, unflagged] = vis
        tb.open(model, nomodify=False)
        tb.putcol("DATA", data)

        tb.flush()
        tb.close()

    # concatenating separated ms files
    concat(vis=model_names, concatvis=target+'_final_data.ms')
