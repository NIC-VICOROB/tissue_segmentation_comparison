import nibabel as nib
import numpy as np

def read_dataset(gen_conf, train_conf) :
    dataset = train_conf['dataset']
    dataset_path = gen_conf['dataset_path']
    dataset_info = gen_conf['dataset_info'][dataset]

    if dataset == 'iSeg2017' :
        return read_iSeg2017_dataset(dataset_path, dataset_info)
    if dataset == 'IBSR18' :
        return read_IBSR18_dataset(dataset_path, dataset_info)
    if dataset == 'MICCAI2012' :
        pass

def read_iSeg2017_dataset(dataset_path, dataset_info) :
    num_volumes = dataset_info['num_volumes']
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, modalities) + dimensions)
    labels = np.zeros((num_volumes, 1) + dimensions)

    for img_idx in range(num_volumes) :
        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[0])
        image_data[img_idx, 0] = read_volume(filename)[:, :, :, 0]
        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        image_data[img_idx, 1] = read_volume(filename)[:, :, :, 0]

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[2])
        labels[img_idx, 0] = read_volume(filename)[:, :, :, 0]

    label_mapper = {0 : 0, 10 : 1, 150 : 2, 250 : 3}
    for key in label_mapper.keys() :
        labels[labels == key] = label_mapper[key]

    return image_data, labels

def read_IBSR18_dataset(dataset_info) :
    num_volumes = dataset_info['num_volumes']
    dimensions = dataset_info['dimensions']
    modalities = dataset_info['modalities']
    dataset_path = gen_conf['dataset_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    image_data = np.zeros((num_volumes, modalities) + dimensions)
    labels = np.zeros((num_volumes, 1) + dimensions)

    for img_idx in range(num_volumes) :
        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[0])
        image_data[img_idx, 0] = read_volume(filename)

        filename = dataset_path + path + pattern.format(img_idx + 1, inputs[1])
        labels[img_idx, 0] = read_volume(filename)

    return image_data, labels

def save_volume(gen_conf, train_conf, volume, case_idx) :
    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step_test']
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    inputs = dataset_info['inputs']

    if dataset == 'iSeg2017' :
        volume_tmp = np.zeros(volume.shape + (1, ))
        volume_tmp[:, :, :, 0] = volume
        volume = volume_tmp

        label_mapper = {0 : 0, 1 : 10, 2 : 150, 3 : 250}
        for key in label_mapper.keys() :
            volume[volume == key] = label_mapper[key]

    data_filename = dataset_path + path + pattern.format(case_idx, inputs[0])
    image_data = read_volume_data(data_filename)

    volume = np.multiply(volume, image_data.get_data() != 0)

    out_filename = results_path + path + pattern.format(case_idx, approach + ' - ' + str(extraction_step))

    __save_volume(volume, image_data, out_filename, dataset_info['format'])

def __save_volume(volume, image_data, filename, format) :
    img = None
    if format == 'nii' :
        img = nib.Nifti1Image(volume, image_data.affine),
    if format == 'analyze' :
        img = nib.analyze.AnalyzeImage(volume.astype('uint8'), image_data.affine)
    nib.save(img, filename)

def read_volume(filename) :
    return read_volume_data(filename).get_data()

def read_volume_data(filename) :
    return nib.load(filename)