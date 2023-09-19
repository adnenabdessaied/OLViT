#!/bin/bash
DATA_FOLDER="../../data/"
python format_data_with_object_descriptions.py \
    --simmc_train_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_train.json" \
    --simmc_dev_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_dev.json" \
    --simmc_devtest_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_devtest.json" \
    --scene_json_folder "/scratch/hochmeister/simmc2/data/public/" \
    --ambiguous_candidates_save_path "/scratch/hochmeister/simmc2/data/subtask_4_b_data/"
    --fashion_prefab_metadata "/scratch/hochmeister/simmc2/data/fashion_prefab_metadata_all.json"
    --furniture_prefab_metadata "/scratch/hochmeister/simmc2/data/furniture_prefab_metadata_all.json"
    --n_answer_candidates 10