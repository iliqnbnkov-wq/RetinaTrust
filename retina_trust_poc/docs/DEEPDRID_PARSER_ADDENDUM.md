# DeepDRiD Parser Compatibility Addendum

Date UTC: 2026-10-24T20:18:50.6216085Z
Original protocol: v0.4.0
Original model: RetinaTrust v0.3

Problem:
Frozen parser accepts only eye image indices 1 and 2.
DeepDRiD also contains indices 3 and 4.

Amendment:
Runtime adapter permits image identifiers l1-l4 and r1-r4.

Frozen model: unchanged
Classification thresholds: unchanged
External training: none
External recalibration: none

This is a documented parser deviation from the frozen implementation.

Adapter SHA256: 23976633D8B340A141F3ED4F2FFC8361CD694017638C6A14ABB52F0A51FE134E
