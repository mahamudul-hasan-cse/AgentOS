from kernel.scheduler import Process, TimeSlice, fcfs, round_robin


def make_processes():
    return [
        Process(pid="P1", arrival_time=0, estimated_burst=5, priority=1),
        Process(pid="P2", arrival_time=1, estimated_burst=3, priority=2),
        Process(pid="P3", arrival_time=2, estimated_burst=8, priority=0),
    ]


def assert_no_gaps_or_overlaps(timeline: list[TimeSlice]) -> None:
    for prev, curr in zip(timeline, timeline[1:]):
        assert curr.start == prev.end, "single CPU: slices must be back-to-back"
        assert curr.start >= prev.start


def assert_conserves_burst(timeline: list[TimeSlice], processes: list[Process]) -> None:
    totals: dict[str, float] = {}
    for slice_ in timeline:
        totals[slice_.pid] = totals.get(slice_.pid, 0.0) + (slice_.end - slice_.start)
    for process in processes:
        assert totals[process.pid] == process.estimated_burst


def test_fcfs_runs_processes_in_arrival_order():
    processes = make_processes()
    timeline = fcfs(processes)

    assert [s.pid for s in timeline] == ["P1", "P2", "P3"]
    assert timeline == [
        TimeSlice(pid="P1", start=0, end=5),
        TimeSlice(pid="P2", start=5, end=8),
        TimeSlice(pid="P3", start=8, end=16),
    ]
    assert_no_gaps_or_overlaps(timeline)
    assert_conserves_burst(timeline, processes)
    assert all(p.state == "terminated" for p in processes)


def test_fcfs_is_non_preemptive_even_with_higher_priority_arrival():
    # priority is irrelevant to FCFS; arrival order alone decides the schedule
    processes = make_processes()
    timeline = fcfs(processes)
    assert timeline[0].pid == "P1"  # earliest arrival, despite P3 having priority 0


def test_round_robin_preempts_on_quantum_and_conserves_burst():
    processes = make_processes()
    timeline = round_robin(processes, quantum=2)

    assert timeline == [
        TimeSlice(pid="P1", start=0, end=2),
        TimeSlice(pid="P2", start=2, end=4),
        TimeSlice(pid="P3", start=4, end=6),
        TimeSlice(pid="P1", start=6, end=8),
        TimeSlice(pid="P2", start=8, end=9),
        TimeSlice(pid="P3", start=9, end=11),
        TimeSlice(pid="P1", start=11, end=12),
        TimeSlice(pid="P3", start=12, end=14),
        TimeSlice(pid="P3", start=14, end=16),
    ]
    assert_no_gaps_or_overlaps(timeline)
    assert_conserves_burst(timeline, processes)
    assert all(p.state == "terminated" for p in processes)

    # no slice should exceed the quantum
    assert all(s.end - s.start <= 2 for s in timeline)

    # P1 is preempted at least once: it appears in more than one slice
    assert sum(1 for s in timeline if s.pid == "P1") > 1


def test_round_robin_rejects_non_positive_quantum():
    try:
        round_robin(make_processes(), quantum=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
