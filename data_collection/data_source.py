from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataSource(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        return "Unnamed"

    @abstractmethod
    def get_data(self) -> Optional[pd.DataFrame]:
        # This method must be implemented by subclasses
        pass
