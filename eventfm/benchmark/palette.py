"""Chart theme: a fixed, pre-validated categorical order plus mode surfaces.

Slots are assigned in this order and never cycled. Every figure in the report
caps a single axes at four series, which is what keeps the adjacent-pair
colour-vision separation of this order valid; anything wider is faceted into
small multiples instead of reaching for a ninth hue.

Because three of the light-mode slots sit below 3:1 against the light surface,
every series also carries a distinct marker and a direct end label, so identity
is never encoded by colour alone.
"""

from dataclasses import dataclass
from typing import Dict, List

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    series: List[str]

    def color(self, index: int) -> str:
        return self.series[index % len(self.series)]

    @staticmethod
    def marker(index: int) -> str:
        return MARKERS[index % len(MARKERS)]

    @staticmethod
    def linestyle(index: int) -> str:
        return LINESTYLES[index % len(LINESTYLES)]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#8a8983",
    grid="#e6e5e1",
    axis="#b4b3ad",
    series=CATEGORICAL_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8a8983",
    grid="#333330",
    axis="#5a5a55",
    series=CATEGORICAL_DARK,
)

THEMES: Dict[str, Theme] = {"light": LIGHT, "dark": DARK}
