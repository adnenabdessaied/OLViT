#!/bin/bash
DATA_FOLDER="../../data/"
python format_data.py \
    --simmc_train_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_train.json" \
    --simmc_dev_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_dev.json" \
    --simmc_devtest_json "/scratch/hochmeister/simmc2/data/simmc2.1_dials_dstc11_devtest.json" \
    --scene_json_folder "/scratch/hochmeister/simmc2/data/public/" \
    --ambiguous_candidates_save_path "/scratch/hochmeister/simmc2/data/ambiguous_candidates/"