"""DetectionStrategy ABC -- the common contract every detection strategy
(BlindMulticatalogStrategy today, a future SubtractionStrategy) implements.

`run()`'s return type is deliberately left as `Any`, not `List[Candidate]`.
BlindMulticatalogStrategy.run() returns a `(final_candidates_table,
lightcurves_dict)` shape -- the real pipeline's Table carries columns
(motion/trail features, strategy fields) that don't cleanly map onto
Candidate without either data loss or stuffing everything into
Candidate.features, and a future subtraction-based strategy may have a
different natural feature set that clarifies the right shape. Converting to
a real `List[Candidate]` contract is deferred to a later, deliberate
decision rather than forced now on incomplete information.
"""

from abc import ABC, abstractmethod
from typing import Any


class DetectionStrategy(ABC):
    @abstractmethod
    def run(self, detection_tables, config) -> Any:
        ...
