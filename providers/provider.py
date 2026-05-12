from abc import ABC, abstractmethod

class Provider(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        raise NotImplementedError("Provider name not defined for provider ", self.__name__)

    @property
    @abstractmethod
    def description(self) -> str:
        """Provider name"""
        raise NotImplementedError("Provider descripiton not defined for provider ", self.__name__)

    @property
    @abstractmethod
    def slug(self) -> str:
        """Provider name"""
        raise NotImplementedError("Unique provider slug not defined for provider ", self.__name__)

    
