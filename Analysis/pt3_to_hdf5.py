import os
import numpy as np
import phconvert as phc
# from pathlib import Path

def pt3_to_hdf5(filename, print_exist=True):
    """
    Arguments:
    filename: .pt3 filename if it is in the same folder or the total path to the .pt3 file
    Returns:
    Nothing. It creates a file with the same name but with hdf5 extension
    """
    file_path = os.path.abspath(filename)
    if not os.path.isfile(file_path):
        print('pt3 file not found')
    file_path_hdf5 = file_path[:-3]+'hdf5'
    if os.path.isfile(file_path_hdf5) and print_exist:
        print(filename[:-3]+'hdf5 already exists')
        #break
    elif os.path.isfile(file_path_hdf5):
        x=0;
    else:
        d, meta = phc.loader.nsalex_pt3(filename,
                                        donor = 1,
                                        acceptor = 0,
                                        alex_period_donor = (4000, 5000),
                                        alex_period_acceptor = (0, 3000),
                                        excitation_wavelengths = (470e-9, 635e-9),
                                        detection_wavelengths = (525e-9, 690e-9),
                                        )
        #Removing the outflow counts
        nanotimes = d['photon_data']['nanotimes']
        detectors = d['photon_data']['detectors']
        timestamps = d['photon_data']['timestamps']

        overflow_nanotimes = d['photon_data']['nanotimes'] != 0

        detectors = detectors[overflow_nanotimes]
        timestamps = timestamps[overflow_nanotimes]
        nanotimes = nanotimes[overflow_nanotimes]
        # Replacing in the dictionary or to be saved variables
        d['photon_data']['nanotimes'] = nanotimes
        d['photon_data']['detectors'] = detectors
        d['photon_data']['timestamps'] = timestamps
        #Metadata
        author = 'Biswajit'
        author_affiliation = 'Leiden University'
        description = 'A demonstrative pt3 data readin.'
        sample_name = 'ttttt'
        dye_names = 'ATTO655'
        buffer_name = 'HEPES pH7 with 100 mM NaCl'
        #Add meta data
        d['description'] = description
        d['sample'] = dict(
            sample_name=sample_name,
            dye_names=dye_names,
            buffer_name=buffer_name,
            num_dyes = len(dye_names.split(',')))
        d['identity'] = dict(
            author=author,
            author_affiliation=author_affiliation)
        _ = meta.pop('dispcurve', None)
        _ = meta.pop('imghdr', None)

        d['user'] = {'picoquant': meta}
        #Save to Phton-HDF5
        phc.hdf5.save_photon_hdf5(d, overwrite=True)
    return(file_path_hdf5)

def t3r_to_hdf5(filename):
    """
    Arguments:
    filename: .pt3 filename if it is in the same folder or the total path to the .pt3 file
    Returns:
    Nothing. It creates a file with the same name but with hdf5 extension
    """
    file_path = os.path.abspath(filename)
    if not file_path.is_file():
        print('t3r file not found')
    file_path_hdf5 = Path(filename[:-3]+'hdf5')
    if file_path_hdf5.is_file():
        print(filename[:-3]+'hdf5 already exists')
        #break
    else:
        d, meta = phc.loader.nsalex_t3r(filename,
                                        donor = 1,
                                        acceptor = 0,
                                        alex_period_donor = (3000, 4000),
                                        alex_period_acceptor = (0, 2000),
                                        excitation_wavelengths = (470e-9, 635e-9),
                                        detection_wavelengths = (525e-9, 690e-9),
                                        )
        #Metadata
        author = 'Biswajit'
        author_affiliation = 'Leiden University'
        description = 'Transient binding on a gold nanorod.'
        sample_name = 'AuNR_DocDNA'
        dye_names = 'ImagCy5'
        buffer_name = 'HEPES pH7 with 100 mM NaCl'
        #Add meta data
        d['description'] = description
        d['sample'] = dict(
            sample_name=sample_name,
            dye_names=dye_names,
            buffer_name=buffer_name,
            num_dyes = len(dye_names.split(',')))
        d['identity'] = dict(
            author=author,
            author_affiliation=author_affiliation)
        _ = meta.pop('dispcurve', None)
        _ = meta.pop('imghdr', None)

        d['user'] = {'picoquant': meta}
        #Save to Phton-HDF5
        phc.hdf5.save_photon_hdf5(d, overwrite=True)
    return(file_path_hdf5)

def pt3t3r_to_hdf5_folder(folderpath, to_hdf5=True, remove_hdf5=False):
    """
    Arguments:
    folderpath:  Give the full path of the folder
    Returns:
    Nothing but saves the files in hdf5 format 
    """
    pt3_extension = [".pt3"]
    t3r_extension = [".t3r"]
    #pt3 conversion
    for dirpath, dirname, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if f.endswith(tuple(pt3_extension))]:
            file_path = os.path.join(dirpath, filename)
            if to_hdf5:
                try:
                    pt3_to_hdf5(filename=file_path, print_exist=False)
                except:
                    print(file_path+'   : is corrupted')
                    pass
    #t3r conversion
    for dirpath, dirname, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if f.endswith(tuple(t3r_extension))]:
            file_path = os.path.join(dirpath, filename)
            if to_hdf5:
                t3r_to_hdf5(filename=file_path)
            if remove_hdf5:
                hdf5_file = file_path[:-3]+'hdf5' 
                if os.path.isfile(hdf5_file):
                    os.remove(hdf5_file)
    return