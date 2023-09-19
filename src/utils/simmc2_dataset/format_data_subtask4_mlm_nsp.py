#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Reads SIMMC 2.1 dataset, creates train, devtest, dev formats for ambiguous candidates.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import json
import os
import random


SPLITS = ["train", "dev", "devtest", "teststd"]


def get_image_name(scene_ids, turn_ind):
    """Given scene ids and turn index, get the image name.
    """
    sorted_scene_ids = sorted(
        ((int(key), val) for key, val in scene_ids.items()),
        key=lambda x: x[0],
        reverse=True
    )
    # NOTE: Hardcoded to only two scenes.
    if turn_ind >= sorted_scene_ids[0][0]:
        scene_label = sorted_scene_ids[0][1]
    else:
        scene_label = sorted_scene_ids[1][1]
    image_label = scene_label
    if "m_" in scene_label:
        image_label = image_label.replace("m_", "")
    return f"{image_label}.png", scene_label


def get_object_mapping(scene_label, args):
    """Get the object mapping for a given scene.
    """
    scene_json_path = os.path.join(
        args["scene_json_folder"], f"{scene_label}_scene.json"
    )
    with open(scene_json_path, "r") as file_id:
        scene_objects = json.load(file_id)["scenes"][0]["objects"]
    object_map = [ii["index"] for ii in scene_objects]
    return object_map


def dictionary_to_string(dictionary):
    result = ""
    for k, v in dictionary.items():
        result += k + ":"
        result += str(v) + " "
    return result


def get_all_answers(dialogs):
    all_answers = []
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        for turn_ind, turn_datum in enumerate(dialog_datum["dialogue"]):
            all_answers.append(turn_datum["system_transcript"])
    return all_answers


def main(args):
    for split in SPLITS:
        read_path = args[f"simmc_{split}_json"]
        print(f"Reading: {read_path}")
        with open(read_path, "r") as file_id:
            dialogs = json.load(file_id)

        # get all answer fromm all dialogues to sample answer candidates from for each dialogue iteration
        all_answers = get_all_answers(dialogs)
        
        ambiguous_candidates_data = []
        for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
            q_turns = []
            a_turns = []
            qa_turns = []

            for turn_ind, turn_datum in enumerate(dialog_datum["dialogue"]):
                query = turn_datum["transcript"]
                answer = turn_datum["system_transcript"]
                
                # wrong answer is used to create false sample for nsp
                wrong_answer = random.choice(all_answers)

                qa_pair = query + '[SEP_1]' + answer + '[SEP]'
                wrong_qa_pair = query + '[SEP_1]' + wrong_answer + '[SEP]'

                image_name, scene_id = get_image_name(
                    dialog_datum["scene_ids"], turn_ind
                )

                # load the scene files and get all the prefab pahts to get the object descriptions for each scene
                prefab_paths = []
                scene_path = os.path.join(args["scene_json_folder"], f"{scene_id}_scene.json")
                with open(scene_path, "r") as scene_file:
                    scene_data = json.load(scene_file)
                for scene in scene_data["scenes"]:
                    for object in scene["objects"]:
                        prefab_paths.append(object["prefab_path"])
                
                # for each dialogue round add a sample with the correct answer and one with a random answer for nsp
                new_datum_correct_answer = {
                    "query": [query],
                    "answer": [answer],
                    "qa_pair": [qa_pair],
                    "next_sentence_label": [1],
                    "q_turns": copy.deepcopy(q_turns),
                    "a_turns": copy.deepcopy(a_turns),
                    "qa_turns": copy.deepcopy(qa_turns),
                    "dialog_id": dialog_datum["dialogue_idx"],
                    "turn_id": turn_ind,
                    "image_name": image_name,
                }
                new_datum_wrong_answer = {
                    "query": [query],
                    "answer": [wrong_answer],
                    "qa_pair": [wrong_qa_pair],
                    "next_sentence_label": [0],
                    "q_turns": copy.deepcopy(q_turns),
                    "a_turns": copy.deepcopy(a_turns),
                    "qa_turns": copy.deepcopy(qa_turns),
                    "dialog_id": dialog_datum["dialogue_idx"],
                    "turn_id": turn_ind,
                    "image_name": image_name,
                }

                ambiguous_candidates_data.append(new_datum_correct_answer)

                if args['create_false_samples_for_nsp']:                
                    ambiguous_candidates_data.append(new_datum_wrong_answer)  

                q_turns.append([query])
                a_turns.append([answer])
                qa_turns.append([qa_pair])


                # Ignore if system_transcript is not found (last round teststd).
                # if turn_datum.get("system_transcript", None):
                #    history.append(turn_datum["system_transcript"])

        print(f"# instances [{split}]: {len(ambiguous_candidates_data)}")
        save_path = os.path.join(
            args["ambiguous_candidates_save_path"],
            f"simmc2.1_ambiguous_candidates_dstc11_{split}.json"
        )
        print(f"Saving: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(
                {
                    "source_path": read_path,
                    "split": split,
                    "data": ambiguous_candidates_data,
                },
                file_id
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simmc_train_json", default=None, help="Path to SIMMC 2.1 train"
    )
    parser.add_argument(
        "--simmc_dev_json", default=None, help="Path to SIMMC 2.1 dev"
    )
    parser.add_argument(
        "--simmc_devtest_json", default=None, help="Path to SIMMC 2.1 devtest"
    )
    parser.add_argument(
        "--simmc_teststd_json", default=None, help="Path to SIMMC 2.1 teststd (public)"
    )
    parser.add_argument(
        "--scene_json_folder", default=None, help="Path to SIMMC scene jsons"
    )
    parser.add_argument(
        "--ambiguous_candidates_save_path",
        required=True,
        help="Path to save SIMMC disambiguate JSONs",
    )
    parser.add_argument(
        "--fashion_prefab_metadata", required=True,
        help="Path to the file with all metadata for fashion objects"
    )
    parser.add_argument(
        "--furniture_prefab_metadata", required=True,
        help="Path to the file with all metadata for fashion objects"
    )
    parser.add_argument(
        "--create_false_samples_for_nsp", action='store_true',
        help="if set, for each correct sample a wrong one is added"
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
