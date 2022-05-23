from abc import ABC, abstractmethod
from typing import Dict, Any

from convSearchPython.pipelines import Pipeline


class PipelineFactory(ABC):
    """
    Interface for a Pipeline factory.

    Implementing classes should override the `set` and `build` methods.

    ## Factory

    A pipeline factory provides a way to compose a pipeline from the run configuration file (in INI format).

    A class implementing this interface should accept key-value pairs extracted directly from the INI file,
    parse them according to its internal rules and build a Pipeline (or Pipeline-like) object.

    When created, a factory receive a dict of arguments: these are common arguments (the same cited in `Pipeline`)
    that should be used to initialize the various steps (in combination eventual additional arguments that might
    be passed to set). For more information on reserved arguments see `convSearchPython.search`.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: common arguments that will be used in step initialization
        """
        pass

    @abstractmethod
    def set(self, config: Dict[str, Any]):
        """
        Set parameters of the factory.

        If the INI file contains:

        ```
        factory_name.param1 = value1
        factoty_name.param2 = value2
        ```

        this method will be called with:

        ```
        factory.set({
            'param1': value1,
            'param2': value2,
        })
        ```

        **Note:** type of values will be guessed (and cast) before passing them to this method

        Args:
            config: dict of parameters for factory configuration
        """
        pass

    @abstractmethod
    def build(self) -> Pipeline:
        """
        Build the pipeline

        Returns:
            A Pipeline object
        """
        pass
