"""LP lock lifecycle management (SPEC §4).

`schedules` is pure math + the per-LP vesting state machine; `locks` wraps the
conviction v2 extrinsics; `monitor` polls chain state for invariant violations.
"""
