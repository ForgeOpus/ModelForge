from __future__ import annotations

import time

from workflow.models import TrainingEvent


def test_training_event_timestamp_is_unix_seconds():
    before = time.time()
    event = TrainingEvent(run_id=1, type="log", message="step")
    after = time.time()

    assert before <= event.ts <= after
    # Millisecond timestamps are about 1e12; the API contract is seconds.
    assert event.ts < 10**11
    description = TrainingEvent.model_fields["ts"].description or ""
    assert "seconds" in description
    assert "milliseconds" in description
