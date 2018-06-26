import os
import sys
import numpy as np
import cv2
import math
import itertools


class RandomizeKFoldDataGeneratorPairGenerator:
    def get_training_and_validation_data_generator_pair(
            self, num_of_folds: int, data_source_dir: str,
            batch_size_of_training_data_set: int, batch_size_of_validation_data_set: int
    ):
        all_data_files = np.array(DataFileListGetter().execute(data_source_dir))
        validation_data_files_within_every_fold = self.__get_split_data_files(all_data_files, num_of_folds)

        for validation_data_files in validation_data_files_within_every_fold:
            training_data_files = np.setdiff1d(all_data_files, validation_data_files)
            training_data_generator = DataGenerator(training_data_files, batch_size_of_training_data_set)

            validation_data_generator = DataGenerator(validation_data_files, batch_size_of_validation_data_set)

            print("training: {}, testing: {}".format(training_data_files.size, validation_data_files.size))
            yield training_data_generator, validation_data_generator

    @staticmethod
    def __get_split_data_files(data_files, num_of_folds):
        size_of_data_within_single_fold = int(math.floor(data_files.size / num_of_folds))
        temp_data_files = data_files
        split_data_files = list()

        for _ in range(num_of_folds - 1):
            selected_data = np.random.choice(temp_data_files, size=size_of_data_within_single_fold, replace=False)
            split_data_files.append(selected_data)
            temp_data_files = np.setdiff1d(temp_data_files, selected_data)

        split_data_files.append(temp_data_files)

        return split_data_files


class SimpleDataGeneratorGetter:
    @staticmethod
    def get_generator(data_source_dir, batch_size):
        data_file_list = DataFileListGetter().execute(data_source_dir)
        return DataGenerator(data_file_list, batch_size)


class DataGenerator:
    def __init__(self, data_file_list, batch_size):
        self.__data_file_list = data_file_list
        self.__batch_size = batch_size

    def infinitely_generate_batch_of_data_pair_tuple(self):
        file_to_convert = list()
        for file in itertools.cycle(self.__data_file_list):
            if len(file_to_convert) < self.__batch_size:
                file_to_convert.append(file)
            else:
                pre_processed_data = DataConverter().read_data_then_expand_and_standardize(file_to_convert)
                yield pre_processed_data, pre_processed_data
                file_to_convert = list()

    def generate_batch_of_data_pair_tuple(self):
        for start_index in range(0, len(self.__data_file_list), self.__batch_size):
            end_index = start_index + self.__batch_size
            batch_of_data_file_list = self.__data_file_list[start_index:end_index]

            pre_processed_data = DataConverter().read_data_then_expand_and_standardize(batch_of_data_file_list)

            yield pre_processed_data, pre_processed_data


class DataFileListGetter:
    @staticmethod
    def execute(data_source_dir: str) -> list:
        return [
            os.path.join(data_source_dir, file)
            for file in os.listdir(data_source_dir)
            if os.path.isfile(
                os.path.join(data_source_dir, file)
            )
        ]


class DataConverter:
    def read_data_then_expand_and_standardize(self, data_file_list: list) -> np.ndarray:
        data_list = self.__fetch_all_data_from_disk(data_file_list)
        converted_data_list = self.__expand_and_standardize(data_list)

        return converted_data_list

    @staticmethod
    def __fetch_all_data_from_disk(data_file_list: list) -> np.ndarray:
        data_list = list()

        for data_file in data_file_list:
            data_list.append(
                np.array(
                    cv2.imread(data_file)
                )
            )

        return np.array(data_list)

    @staticmethod
    def __expand_and_standardize(data_list: np.ndarray) -> np.ndarray:
        converted_data_list = data_list.astype(np.float32)
        converted_data_list = converted_data_list/255

        return converted_data_list


class ApplicationDirPathGetter:
    @staticmethod
    def execute() -> str:
        if getattr(sys, 'frozen', False):
            application_path = sys.executable

        elif hasattr(sys.modules['__main__'], "__file__"):
            application_path = os.path.abspath(sys.modules['__main__'].__file__)

        else:
            application_path = sys.executable

        return os.path.dirname(application_path)

