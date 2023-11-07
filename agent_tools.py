import os
from math import (
    sqrt,
    cos,
    sin,
)
from typing import (
    Any,
    Optional,
    Union,
)

from alpha_vantage.timeseries import TimeSeries
from langchain.tools import BaseTool


class StockTool(BaseTool):
    name = "Stock Quote Tool"
    description = ("use this tool to find stock quotes. To use the tool you must provide the "
                   "stock_ticker_symbol argument. This is the"
                   "ticker symbol of the stock for which you need a quote.")

    def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async run not implemented yet...")

    def _run(self, stock_ticker_symbol):
        #stock = yf.Ticker(stock_ticker_symbol)
        #return stock.info
        ts = TimeSeries(key=os.environ.get('ALPHA_VANTAGE_API_KEY'))
        quote = ts.get_quote_endpoint(stock_ticker_symbol)
        return quote


class DirectoryTool(BaseTool):
    name = "Directory Tool"
    description = "use this tool to find json data about a person in the directory"

    def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async run not implemented yet...")

    def _run(self, person_name):
        directory = {
            "Glenn": {
                "age": 32,
                "occupation": "engineer",
                "location": "Fishers, IN, USA",
                "hobbies": ["hiking", "biking", "skiing"],
            },
            "Jane": {
                "age": 28,
                "occupation": "data scientist",
                "hobbies": ["painting", "cooking", "reading"],
            },
            "Bob": {
                "age": 45,
                "occupation": "manager",
                "hobbies": ["fishing", "hunting", "hiking"],
                "contact_info": {
                    "email": "bob@bob.com",
                    "phone": "555-555-5555",
                    "address": "123 Main St.",
                },
            },
        }
        return directory.get(person_name, f"I do not know {person_name}.")


class PythogorasTool(BaseTool):
    name = "Hypotenuse Calculator"
    description = """use this tool when you need to calculate the length of an hypotenuse of a right triangle given one or two sides of a triangle and/or an angle (in degrees).
    To use the tool you must provide at least two of the following parameters: ['adjacent_side', 'opposite_side', 'angle'].
    """

    def _run(
        self,
        adjacent_side: Optional[Union[int, float]] = None,
        opposite_side: Optional[Union[int, float]] = None,
        angle: Optional[Union[int, float]] = None,
    ):
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side) ** 2 + float(opposite_side) ** 2)
        elif adjacent_side and angle:
            return float(adjacent_side) / cos(float(angle))  # * pi / 180)
        elif opposite_side and angle:
            return float(opposite_side) / sin(float(angle))
        else:
            return "Could not calculate hypotenuse."

    def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("async run not implemented yet...")


class DoublerTool(BaseTool):
    name = "DoublerTool"
    description = "a tool for doubling numbers"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run(self, message: Union[int, float]):
        return (float(message) * 2)

    def _arun(self, message):
        raise NotImplementedError("async run not implemented yet...")
