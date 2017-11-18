general_configuration = {
    'dataset_path' : 'datasets',
    'log_path' : 'log',
    'model_path' : 'models',
    'results_path' : 'results',
    'dataset_info' : {
        'iSeg2017' : {
            'num_volumes' : 10,
            'general_pattern' : 'subject-{}-{}.hdr',
            'path' : 'iSeg2017/iSeg-2017-Training'
        },
        'IBSR18' : {
            'num_volumes' : 18,
            'general_pattern' : 'IBSR_{0:02}/IBSR_{0:02}_{}.nii.gz',
            'path' : 'IBSR18'
        },
        'MICCAI2012' : {
            'num_volumes' : [15, 20],
            'general_pattern' : ['{}/1{:02}.nii.gz', '{}/1{:02}_3C.nii.gz'],
            'folder_names' : ['training_images', 'training-labels', 'testing-images', 'testing-labels'],
            'path' : 'MICCAI2012'
        }
    }
}

training_configuration = {
    'dataset' : 'iSeg2017',
    'approach' : 'DolzMulti',
    'patch_shape' : (27, 27, 27),
    'output_shape' : (9, 9, 9),
    'extraction_step' : (9, 9, 9),
    'num_epochs' : 20,
    'patience' : 1,
    'validation_split' : 0.20,
    'verbose' : 1
}