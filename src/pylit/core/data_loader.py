import os
import csv
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from pylit.global_settings import ARRAY


class DataLoader:
    _data: ARRAY = None

    def __init__(
        self,
        file_path: str,
        header: Optional[List[str]] = None,
    ) -> None:
        """DataLoader class to load data from a file and store it in a numpy array.

        Args:
        -----
            file_path (str):
                Path to the data file.
            header (List[str]):
                List of headers to be used for the data."""

        self._header = header
        self._file_path = file_path

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str) -> None:
        if not isinstance(file_path, str):
            raise TypeError("File path must be a string.")
        if not os.path.isfile(file_path):
            raise ValueError(f"File: {file_path} does not exist.")
        self._file_path = file_path

    @property
    def header(self) -> ARRAY:
        return self._header

    @header.setter
    def header(self, header: List[str]) -> None:
        if not isinstance(header, list):
            raise TypeError("Header must be a list.")
        if not all(isinstance(head, str) for head in header):
            raise TypeError("Header must be a list of strings.")
        self._header = header

    @property
    def data(self) -> ARRAY:
        return self._data

    @data.setter
    def data(self, data: ARRAY) -> None:
        raise AttributeError("Cannot change the data attribute directly.")

    def fetch(self) -> None:
        """
        Method to fetch the data from the desired file and store it in the data property. The data functionates as a stack.

        Raises:
            ValueError:
                If file path has not been provided.
        """
        if self._file_path is None:
            raise ValueError("File path has not been provided.")

        if self._header is None:
            self._data = np.genfromtxt(self._file_path)
        else:
            self._data = np.genfromtxt(
                self._file_path,
                names=self._header,
            )

    def clear(self) -> None:
        """Method to clear the attributes stored in the data property."""

        self._file_path = None
        self._header = None
        self._data = None

    def __call__(self, *headers: Union[str, int]) -> Tuple:
        """Method to get data by headers.

        Args:
        -----
            headers (Union[str, int]):
                Headers to get the data from.

        Raises:
        -------
            ValueError:
                If data has not been fetched yet.
            ValueError:
                If invalid header is provided.

        Returns:
        --------
            Tuple:
                Tuple of data by headers."""

        if self._data is None:
            raise ValueError("Data has not been fetched yet.")

        if self._header is not None:
            if any(header not in self._header for header in headers):
                raise ValueError("Invalid header provided.")
            return tuple([self._data[header] for header in headers])
        else:
            return tuple([self._data.T[header] for header in headers])

    def store(self, file_path: str, *headers: Union[str, int]):
        """
        Stores a subset of the data corresponding to the given headers into a CSV file.

        Parameters:
        -----------
            file_path : str
                The path to the CSV file where the data will be stored.
            headers : Union[str, int]
                The headers for the subset of data to be stored.

        Raises:
        -------
            ValueError:
                If data has not been fetched yet.
                If invalid header is provided.
        """
        data_to_store = self(*headers)
        # Writing to CSV
        with open(file_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in zip(*data_to_store):
                csvwriter.writerow(row)
