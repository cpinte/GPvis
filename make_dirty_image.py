#execfile('reduction_utils.py')
import casatasks
import os


def make_dirty_image(ms_file):

    threshold = '0.00mJy'
    scales = [0, 20, 50, 100, 200]

    imagename = ms_file.replace(".ms","")


    ext = [".image",".psf",".pb",".residual",".sumwt",".model",".fits"]
    for e in ext:
        os.system("rm -rf "+imagename+e)

    casatasks.tclean(vis= ms_file,
                     imagename = imagename,
                     specmode = 'mfs',
                     deconvolver = 'multiscale',
                     scales = scales,
                     weighting='briggs',
                     robust = -2,
                     gain = 0.3,
                     imsize = 1000,
                     cell = '.003arcsec',
                     smallscalebias = 0.6, #set to CASA's default of 0.6 unless manually changed
                     niter = 0, # Ditry image
                     interactive = False,
                     threshold = threshold,
                     cycleniter = 300,
                     cyclefactor = 1,
                     uvtaper = [],
                     mask = '',
                     savemodel = 'none',
                     nterms = 1)

    casatasks.exportfits(imagename+'.image', imagename+'.fits')




ms_files = ["HD143006_final_data_spavg_split_7_updated.ms","HD143006_final_data_spavg_split_7.ms"]

ms_files = ["HD143006_final_data_spavg.ms","HD143006_final_data_spavg_updated.ms"]


for ms_file in ms_files:
    make_dirty_image(ms_file)
