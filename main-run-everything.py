from configuration import general_configuration, training_configuration
from workflow.evaluate import run_evaluation_in_dataset

import numpy as np
import nibabel as nib

def calculate_dice(general_configuration, training_configuration) :
    gen_conf = general_configuration
    train_conf = training_configuration

    dataset = train_conf['dataset']
    approach = train_conf['approach']
    extraction_step = train_conf['extraction_step_test']  
    dataset_info = gen_conf['dataset_info'][dataset]
    dataset_path = gen_conf['dataset_path']
    results_path = gen_conf['results_path']
    num_classes = 4
    path = dataset_info['path']
    pattern = dataset_info['general_pattern']
    num_volumes = dataset_info['num_volumes']

    if dataset == 'IBSR18' or dataset == 'iSeg2017' :
        inputs = dataset_info['inputs']
    
        dsc_vals = np.zeros((num_volumes, num_classes))
        for img_idx in range(num_volumes) :
            filename_gt = dataset_path + path + pattern.format(img_idx + 1, inputs[-1])
            filename_seg = results_path + path + pattern.format(img_idx + 1, approach + ' - ' + str(extraction_step))

            gt_vol = nib.load(filename_gt).get_data()
            seg_vol = nib.load(filename_seg).get_data()
            for it, key in enumerate(np.unique(gt_vol)) :
                binary_gt = gt_vol == key
                binary_seg = seg_vol == key
                intersection = np.sum(np.multiply(binary_gt, binary_seg))
                union = np.sum(binary_gt) + np.sum(binary_seg)

                dsc_vals[img_idx, it] = 2.0 * intersection / union

    else :
        folder_names = dataset_info['folder_names']

        testing_set = [1003, 1019, 1038, 1107, 1119, 1004, 1023, 1039, 1110, 1122, 1005,
            1024, 1101, 1113, 1125, 1018, 1025, 1104, 1116, 1128]

        dsc_vals = np.zeros((num_volumes[1], num_classes))
        for img_idx in range(num_volumes[1]) :
            case_idx = testing_set[img_idx]
            filename_gt = dataset_path + path + pattern[1].format(folder_names[3], case_idx)
            filename_seg = results_path + path + pattern[2].format(folder_names[3], case_idx, approach + ' - ' + str(extraction_step))

            gt_vol = nib.load(filename_gt).get_data()
            seg_vol = nib.load(filename_seg).get_data()

            gt_vol[gt_vol > 4] = 0
            for it, key in enumerate(np.unique(gt_vol)) :
                binary_gt = gt_vol == key
                binary_seg = seg_vol == key
                intersection = np.sum(np.multiply(binary_gt, binary_seg))
                union = np.sum(binary_gt) + np.sum(binary_seg)

                dsc_vals[img_idx, it] = 2.0 * intersection / union

    print dsc_vals
    for a_class in range(1, num_classes) :
        print ', '.join(str(x) for x in dsc_vals[:, a_class])

    print np.mean(dsc_vals[:, 1:], axis=(0)), np.std(dsc_vals[:, 1:], axis=(0))

number_of_skips = 23

for dimension in [2, 3] :
    training_configuration['dimension'] = dimension
    for approach in ["Guerrero", "Cicek", "DolzMulti", "Kamnitsas"] :
        training_configuration['approach'] = approach
        
        extraction_steps = ()
        extraction_step_tests = ()

        if approach == "DolzMulti" :
            training_configuration['output_shape'] = tuple(9 for i in range(dimension))
            training_configuration['patch_shape'] = tuple(27 for i in range(dimension))

            add = (3, ) if dimension - 2 == 1 else ()

            # 0%, 52%, 88%
            extraction_steps = [tuple(9 for i in range(dimension)), tuple(7 for i in range(dimension)), tuple(5 for i in range(dimension))]
            extraction_step_tests = [tuple(9 for i in range(dimension)), tuple(7 for i in range(dimension)), (3, 9) + add]
        
        if approach == "Kamnitsas" :
            training_configuration['output_shape'] = tuple(16 for i in range(dimension))
            training_configuration['patch_shape'] = tuple(48 for i in range(dimension))

            # 0%, 47%, 87.5%
            extraction_steps = [tuple(16 for i in range(dimension)), tuple(13 for i in range(dimension)), tuple(10 for i in range(dimension))]
            extraction_step_tests = [tuple(16 for i in range(dimension)), tuple(13 for i in range(dimension)), tuple(8 for i in range(dimension))]

        if approach == "Guerrero" or approach == "Cicek" :
            training_configuration['output_shape'] = tuple(32 for i in range(dimension))
            training_configuration['patch_shape'] = tuple(32 for i in range(dimension))

            # 0%, 47%, 94.7%
            extraction_steps = [tuple(32 for i in range(dimension)), tuple(25 for i in range(dimension)), tuple(12 for i in range(dimension))]
            extraction_step_tests = [tuple(32 for i in range(dimension)), tuple(25 for i in range(dimension)), tuple(12 for i in range(dimension))]

        print approach, extraction_steps, extraction_step_tests

        for extraction_step in extraction_steps :
            if number_of_skips != 0 :
                number_of_skips -= 1
                continue

            training_configuration['num_epochs'] = 20
        
            training_configuration['extraction_step'] = extraction_step
            training_configuration['extraction_step_test'] = extraction_step_tests[0]
            run_evaluation_in_dataset(general_configuration, training_configuration)

            print approach, extraction_step, extraction_step_tests[0]
            calculate_dice(general_configuration, training_configuration)

            training_configuration['num_epochs'] = 0
            training_configuration['extraction_step_test'] = extraction_step_tests[1]
            run_evaluation_in_dataset(general_configuration, training_configuration)

            print approach, extraction_step, extraction_step_tests[1]
            calculate_dice(general_configuration, training_configuration)

            training_configuration['extraction_step'] = extraction_step
            training_configuration['extraction_step_test'] = extraction_step_tests[2]
            run_evaluation_in_dataset(general_configuration, training_configuration)

            print approach, extraction_step, extraction_step_tests[2]
            calculate_dice(general_configuration, training_configuration)