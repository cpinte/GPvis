# we denoise 1 execution
denoise_visibilities.denoise_npz("HD143006_final_data_spavg_split_7.vis.npz",plot=True)

# We create a ms file
ms_utils.update_ms_file("./HD143006_final_data_spavg_split_7.ms")

# We make the dirty images
%run -i make_dirty_image.py
