"""Small reusable widgets"""

import math

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox


class ScientificSpinBox(QDoubleSpinBox):
    """
    A spin box for values spanning many orders of magnitude (gas densities ~1e-5 g/cm3):
    shows significant digits instead of fixed decimals, accepts "1.08e-5" and decimal
    commas, and the arrows step by a relative amount (default 1 %).

    A plain QDoubleSpinBox with 8 decimals keeps only 3 significant digits at 1e-5.
    """

    def __init__(self, sig_digits: int = 7, rel_step: float = 0.01, parent=None):
        super().__init__(parent)
        self._sig = sig_digits
        self._rel_step = rel_step
        self.setDecimals(15)             # internal precision: never round a 1e-5 value
        self.setKeyboardTracking(False)  # re-run on Enter / focus-out, not on every keystroke

    def _strip(self, text: str) -> str:
        t = text
        if self.prefix() and t.startswith(self.prefix()):
            t = t[len(self.prefix()):]
        if self.suffix() and t.endswith(self.suffix()):
            t = t[: -len(self.suffix())]
        return t.strip().replace(",", ".")

    def textFromValue(self, value: float) -> str:
        return f"{value:.{self._sig}g}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(self._strip(text))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        t = self._strip(text)
        if t in ("", "-", "+", ".", "e", "E") or t.lower().endswith(("e", "e-", "e+")):
            return QValidator.Intermediate, text, pos
        try:
            v = float(t)
        except ValueError:
            return QValidator.Invalid, text, pos
        if not math.isfinite(v):
            return QValidator.Invalid, text, pos
        if self.minimum() <= v <= self.maximum():
            return QValidator.Acceptable, text, pos
        return QValidator.Intermediate, text, pos

    def fixup(self, text: str) -> str:
        return self.textFromValue(self.value())

    def stepBy(self, steps: int):
        v = self.value()
        if v > 0:
            self.setValue(v * (1.0 + self._rel_step) ** steps)
        else:
            super().stepBy(steps)
