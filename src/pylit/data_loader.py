import os
import numpy as np

from typing import Tuple, List, Optional, Union
from pylit.global_settings import ARRAY
from pylit.ui.settings import PATH_DATA


class DataLoader:
    _data: ARRAY = None

    def __init__(
        self,
        file_name: str,
        directory: str = PATH_DATA,
        header: Optional[List[str]] = None,
    ) -> None:
        """DataLoader class to load data from a file and store it in a numpy array.

        Args:
        -----
            file_name (str):
                Name of the file to load the data from.
            header (List[str]):
                List of headers to be used for the data.
            directory (str):
                Path to the data file."""

        self._header = header
        self.directory = directory
        self.file_name = file_name

    @property
    def file_name(self) -> str:
        """File name."""
        return self._file_name

    @file_name.setter
    def file_name(self, file_name: str) -> None:
        if not isinstance(file_name, str):
            raise TypeError("File name must be a string.")

        full_path = os.path.join(self._directory, file_name)
        if not os.path.isfile(full_path):
            raise ValueError(f"File: {full_path} does not exist.")
        self._file_name = file_name

    @property
    def header(self) -> ARRAY:
        """Header."""
        return self._header

    @header.setter
    def header(self, header: List[str]) -> None:
        if not isinstance(header, list):
            raise TypeError("Header must be a list.")
        if not all(isinstance(head, str) for head in header):
            raise TypeError("Header must be a list of strings.")
        self._header = header

    @property
    def directory(self) -> str:
        """directory."""
        return self._directory

    @directory.setter
    def directory(self, directory: str) -> None:
        if not isinstance(directory, str):
            raise TypeError("Directory must be a string.")
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")
        self._directory = directory

    @property
    def data(self) -> ARRAY:
        """Data."""
        return self._data

    @data.setter
    def data(self, data: ARRAY) -> None:
        raise AttributeError(
            "Cannot change the data attribute directly. Use the fetch method to set the data."
        )

    def fetch(self) -> None:
        """Method to fetch the data from the desired file and store it in the data property.

        Raises:
        -------
            ValueError:
                If file name has not been set yet."""

        if self._file_name is None:
            raise ValueError("File name has not been set yet.")

        full_path = os.path.join(self._directory, self._file_name)
        self._data = (
            np.genfromtxt(
                full_path,
                names=self._header,
            )
            if self._header is not None
            else np.genfromtxt(full_path)
        )

    def clear(self) -> None:
        """Method to clear the attributes stored in the data property."""

        self._file_name = None
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

    def dict(self) -> dict:
        """Method to return the data as a dictionary.

        Raises:
        -------
            ValueError:
                If data has not been fetched yet."""

        if self._data is None:
            raise ValueError("Data has not been fetched yet.")

        return (
            {
                header: {"contents": self._data[header]}
                for header in self._header
            }
            if self._header is not None
            else {
                idx: {"contents": self._data.T[idx]}
                for idx in range(len(self._data.T))
            }
        )


if __name__ == "__main__":
    pass
