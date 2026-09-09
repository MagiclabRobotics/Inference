import time

def _sleep_rate(last_t: float, rate_hz: float) -> float:
    period = 1.0 / max(float(rate_hz), 1e-6)
    now = time.monotonic()
    sleep_s = period - (now - last_t)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return time.monotonic()



def test_frequency():
    for i in range(100):
        t = time.monotonic()
        _t = _sleep_rate(t, 30)
        print(_t - t)
