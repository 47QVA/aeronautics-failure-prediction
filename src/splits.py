"""Unit-level train/val/test split for N-CMAPSS DS01.

NASA ships DS01 pre-split by unit into dev (train pool) and test:
  dev  = units 1-6
  test = units 7-10
This test split is unit-disjoint from dev and is held out untouched until
final model evaluation.

Within dev, each unit has a single fixed flight class (Fc): two units per
class (1, 2, 3). To pick a validation set that doesn't leak units and still
lets both train and val see all three flight classes, we hold out one unit
per class for validation:

  class 1: units {1, 4} -> train 1, val 4
  class 2: units {3, 6} -> train 3, val 6
  class 3: units {2, 5} -> train 2, val 5
"""
TRAIN_UNITS = (1, 2, 3)
VAL_UNITS = (4, 5, 6)
TEST_UNITS = (7, 8, 9, 10)

UNIT_TO_FLIGHT_CLASS = {
    1: 1, 4: 1,
    3: 2, 6: 2,
    2: 3, 5: 3,
    7: 1, 9: 1,
    8: 2,
    10: 3,
}
