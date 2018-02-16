import os
import sys
import numpy as np
import cv2


class DataGenerator:
    def __init__(self, data_source_dir: str, batch_size: int):
        self.__data_source_dir = data_source_dir
        self.__batch_size = batch_size

    def generate_batch_of_data_pair_tuple(self):
        data_file_list = [
            os.path.join(self.__data_source_dir, file)
            for file in os.listdir(self.__data_source_dir)
            if os.path.isfile(
                os.path.join(self.__data_source_dir, file)
            )
        ]
        print(len(data_file_list))

        while True:
            for start_index in range(0, len(data_file_list), self.__batch_size):
                end_index = start_index + self.__batch_size
                batch_of_data_file_list = data_file_list[start_index:end_index]

                pre_processed_data = DataConverter().read_data_then_expand_and_standardize(batch_of_data_file_list)

                yield pre_processed_data, pre_processed_data


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

