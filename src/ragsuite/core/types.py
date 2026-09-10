from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import NewType, List, Optional, Dict

from ragsuite.utilities import err

logger: logging.Logger = logging.getLogger(__name__)

# * --------------- QUERY ---------------

QueryStr = NewType("QueryStr", str)
ResponseStr = NewType("ResponseStr", str)


# * --------------- TRANSLATION ---------------
class TranslationMethod(Enum):
    MULTI_QUERY = "multi-query"
    HYDE = "hyde"
    IDENTITY = "identity"
    STEPBACK = "stepback"
    DECOMPOSITION = "decomposition"


TranslationRoute = NewType("TranslationRoute", List[TranslationMethod])


@dataclass(frozen=True)
class TranslationContext:
    query: QueryStr
    quantity: Optional[int] = None  # for MultiQuery
    max_tokens: Optional[int] = None  # for HyDE

    def to_dict(self):
        return {
            "query": self.query,
            "quantity": self.quantity,
            "max_tokens": self.max_tokens
        }


@dataclass(frozen=True)
class HeuristicAnalysisParameters:
    short_len_le: int = 12  # queries with length <= this are considered short


HeuristicAnalysis = NewType("HeuristicAnalysis", Dict[str, bool])


@dataclass
class QueryList:
    """Stores the list of queries produced by various query translators along with the original query and the route (translation methods used)."""
    original_query: QueryStr
    queries: List[QueryStr]
    route: TranslationRoute = field(default_factory=lambda: TranslationRoute([]))

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return self.queries[index]

    def __eq__(self, other: QueryList):
        if not isinstance(other, QueryList):
            return False
        return (self.original_query == other.original_query and
                self.route == other.route)

    def extend(self, other_list: QueryList):
        if self != other_list:
            err.log_and_raise(
                logger,
                ValueError("Cannot extend QueryList with a different original_query or translation_router"),
            )
        self.queries.extend(other_list.queries)

    def to_dict(self):
        return {
            "original_query": self.original_query,
            "queries": self.queries,
            "route": self.route
        }

    def add_step(self, method: TranslationMethod):
        if not method in self.route:
            self.route.append(method)
        else:
            err.log_and_raise(logger, ValueError(f"Method {method} already in the route: {self.route}"))


# * --------------- FUSION ---------------
@dataclass(frozen=True)
class RRFConfig:
    top_k: int
    k_rrf: int

    def to_dict(self):
        return {
            "top_k": self.top_k,
            "k_rrf": self.k_rrf
        }


# * --------------- INGESTION ---------------
@dataclass(frozen=True)
class SemanticTextSplitterConfig:
    bufsz: int
    breakpoint_percentile_threshold: float

    def to_dict(self):
        return {
            "buffer size": self.bufsz,
            "breakpoint_percentile_threshold": self.breakpoint_percentile_threshold,
        }
