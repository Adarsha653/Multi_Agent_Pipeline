import time

from utils.agent_timing import append_step_duration


def test_append_step_duration_accumulates():
    s: dict = {'agent_steps': []}
    t0 = time.perf_counter()
    time.sleep(0.01)
    u = append_step_duration(s, 'search_agent', t0)
    assert 'agent_steps' in u
    assert len(u['agent_steps']) == 1
    assert u['agent_steps'][0]['agent'] == 'search_agent'
    assert u['agent_steps'][0]['seconds'] >= 0.0

    s2 = {**s, **u}
    t1 = time.perf_counter()
    time.sleep(0.01)
    u2 = append_step_duration(s2, 'analysis_agent', t1)
    assert len(u2['agent_steps']) == 2
    assert u2['agent_steps'][1]['agent'] == 'analysis_agent'
