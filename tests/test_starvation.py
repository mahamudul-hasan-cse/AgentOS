"""Starvation under priority scheduling, and the aging mechanisms that fix it.

Each test states the property it is pinning down, because most of them would
also pass against a broken scheduler if you only looked at averages.
"""

from kernel.scheduler import (
    Process,
    TimeSlice,
    effective_priority,
    mlfq,
    mlfq_boost,
    priority_aging,
    priority_scheduling,
)

# A workload built to starve the bottom of the queue: a stream of short,
# top-priority arrivals whose offered load exceeds 1.0 (burst 3 every 2 time
# units), so the CPU never runs out of priority-0 work, plus one long
# low-priority process that arrives first and then has to compete with it.
STREAM_LENGTH = 30
STREAM_BURST = 3.0
STREAM_INTERARRIVAL = 2.0
VICTIM_PID = "VICTIM"
VICTIM_PRIORITY = 3
VICTIM_BURST = 8.0


def starving_workload() -> list[Process]:
    processes = [
        Process(pid=VICTIM_PID, arrival_time=0.0, estimated_burst=VICTIM_BURST,
                priority=VICTIM_PRIORITY)
    ]
    processes += [
        Process(
            pid=f"H{i:02d}",
            arrival_time=i * STREAM_INTERARRIVAL,
            estimated_burst=STREAM_BURST,
            priority=0,
        )
        for i in range(STREAM_LENGTH)
    ]
    return processes


def waiting_time(timeline: list[TimeSlice], process: Process) -> float:
    """turnaround - CPU time received, computed from the timeline."""
    slices = [s for s in timeline if s.pid == process.pid]
    assert slices, f"{process.pid} never ran"
    completion = max(s.end for s in slices)
    return (completion - process.arrival_time) - process.estimated_burst


def max_gap(timeline: list[TimeSlice], pid: str, arrival: float) -> float:
    """Longest stretch the process sat runnable but off the CPU."""
    slices = sorted((s for s in timeline if s.pid == pid), key=lambda s: s.start)
    assert slices, f"{pid} never ran"
    spans = [slices[0].start - arrival]
    spans += [b.start - a.end for a, b in zip(slices, slices[1:])]
    return max(spans)


def test_priority_scheduling_starves_the_low_priority_process():
    """(1) Plain priority scheduling lets a low-priority process starve.

    The victim arrives first and needs only 8 time units, yet it waits behind
    every single one of the 30 higher-priority arrivals -- roughly the whole
    stream's burst -- because the ready queue is re-examined at every dispatch
    and a newcomer always outranks it.
    """
    processes = starving_workload()
    victim = processes[0]
    timeline = priority_scheduling(processes)

    stream_burst = STREAM_LENGTH * STREAM_BURST  # 90.0
    assert waiting_time(timeline, victim) > 80.0
    # it is not merely late, it is served dead last
    assert max(timeline, key=lambda s: s.end).pid == VICTIM_PID
    # and the wait is the whole stream, not some incidental queueing delay
    assert waiting_time(timeline, victim) >= stream_burst - VICTIM_BURST


def test_aging_bounds_the_wait_on_the_identical_workload():
    """(2) Priority + aging keeps the same victim's wait bounded.

    The bound is structural, not empirical: the victim gains one priority level
    per `aging_interval` waited, so after at most priority * aging_interval it
    reaches the top of the queue and can no longer be passed over by a
    same-priority newcomer (ties break on arrival time, and it arrived first).
    """
    aging_interval = 5.0
    processes = starving_workload()
    victim = processes[0]
    timeline = priority_aging(processes, aging_interval=aging_interval)

    # the analytic bound, plus one dispatch it may have to sit through because
    # this scheduler is non-preemptive
    bound = VICTIM_PRIORITY * aging_interval + STREAM_BURST
    assert waiting_time(timeline, victim) <= bound

    # and the fix is a real improvement, not a re-labelling
    starved = waiting_time(priority_scheduling(starving_workload()), victim)
    assert waiting_time(timeline, victim) < starved / 4


def test_aging_bound_holds_as_the_stream_grows():
    """(2b) The bound is a constant, not just a smaller number.

    Lengthening the stream 6x makes the unaged wait grow with it; the aged wait
    must not move at all. This is what separates "bounded" from "still large".
    """
    aging_interval = 5.0
    unaged, aged = [], []
    for stream_length in (10, 30, 60):
        for algorithm, sink in ((priority_scheduling, unaged), (priority_aging, aged)):
            processes = [
                Process(pid=VICTIM_PID, arrival_time=0.0,
                        estimated_burst=VICTIM_BURST, priority=VICTIM_PRIORITY)
            ] + [
                Process(pid=f"H{i:02d}", arrival_time=i * STREAM_INTERARRIVAL,
                        estimated_burst=STREAM_BURST, priority=0)
                for i in range(stream_length)
            ]
            victim = processes[0]
            kwargs = {"aging_interval": aging_interval} if sink is aged else {}
            sink.append(waiting_time(algorithm(processes, **kwargs), victim))

    assert unaged[-1] > 4 * unaged[0], "unaged wait should track the stream length"
    assert max(aged) == min(aged), "aged wait must be constant in stream length"


def test_aging_does_not_invert_simultaneous_arrivals():
    """(3) Before any aging has occurred, aging changes nothing.

    Two processes that arrive together have waited for exactly the same amount
    of time whenever the scheduler looks at them, so they always receive the
    same boost and their relative order is preserved. The property is checked
    both at the unit level and end-to-end.
    """
    high = Process(pid="HIGH", arrival_time=0.0, estimated_burst=5.0, priority=0)
    low = Process(pid="LOW", arrival_time=0.0, estimated_burst=5.0, priority=3)

    # no time has passed: effective priority is exactly the base priority
    assert effective_priority(high, now=0.0) == 0
    assert effective_priority(low, now=0.0) == 3

    # equal waits give equal boosts, so the ordering survives at every instant
    for now in (0.0, 1.0, 4.9, 5.0, 12.0, 40.0, 1000.0):
        assert effective_priority(high, now) <= effective_priority(low, now)

    # end-to-end: the higher-priority process still runs first
    timeline = priority_aging([low, high])
    assert timeline[0].pid == "HIGH"

    # and aging must never overtake the top priority outright, only tie with it
    starved = Process(pid="S", arrival_time=0.0, estimated_burst=1.0, priority=3)
    assert effective_priority(starved, now=10_000.0) == 0


def test_mlfq_boost_promotes_a_long_waiting_process():
    """(4) The periodic boost returns a demoted process to the top queue.

    Plain MLFQ starves by burst length rather than by declared priority: the
    long process is demoted out of level 0 and the stream of short arrivals
    keeps level 0 non-empty forever. The check is on the largest gap between
    consecutive slices -- the interval during which the process was runnable
    and passed over -- which the boost must cap at roughly one boost period.
    """
    boost_interval = 20.0
    quantums = (4.0, 8.0, 16.0)

    def workload() -> list[Process]:
        return [
            Process(pid="LONG", arrival_time=0.0, estimated_burst=40.0, priority=0)
        ] + [
            Process(pid=f"S{i:02d}", arrival_time=i * 2.0, estimated_burst=2.0, priority=0)
            for i in range(40)
        ]

    plain = mlfq(workload(), quantums=quantums)
    boosted = mlfq_boost(workload(), quantums=quantums, boost_interval=boost_interval)

    plain_gap = max_gap(plain, "LONG", 0.0)
    boosted_gap = max_gap(boosted, "LONG", 0.0)

    # without the boost the demoted process is passed over for a long stretch
    assert plain_gap > 3 * boost_interval
    # with it, never for much more than one boost period (plus the slice that
    # was already running when the boost fired)
    assert boosted_gap <= boost_interval + max(quantums)
    assert boosted_gap < plain_gap

    # the promotion is observable as extra slices: being returned to level 0
    # means being dispatched again rather than waiting out the whole stream
    assert len([s for s in boosted if s.pid == "LONG"]) > len(
        [s for s in plain if s.pid == "LONG"]
    )


def test_boost_interval_is_a_dial_not_a_switch():
    """Shorter boost periods bound the gap more tightly. Monotonic, so the
    parameter behaves like the fairness dial it is documented to be."""
    quantums = (4.0, 8.0, 16.0)

    def workload() -> list[Process]:
        return [
            Process(pid="LONG", arrival_time=0.0, estimated_burst=40.0, priority=0)
        ] + [
            Process(pid=f"S{i:02d}", arrival_time=i * 2.0, estimated_burst=2.0, priority=0)
            for i in range(40)
        ]

    gaps = [
        max_gap(
            mlfq_boost(workload(), quantums=quantums, boost_interval=interval),
            "LONG",
            0.0,
        )
        for interval in (10.0, 20.0, 40.0, 80.0)
    ]
    assert gaps == sorted(gaps), f"gap should grow with the boost interval: {gaps}"


def test_aging_variants_preserve_total_work():
    """Neither fix may invent or lose CPU time: both are work-conserving, and
    every process must still receive exactly its full burst."""
    for timeline, processes in (
        (priority_aging(starving_workload()), starving_workload()),
        (mlfq_boost(starving_workload()), starving_workload()),
    ):
        received: dict[str, float] = {}
        for slice_ in timeline:
            received[slice_.pid] = received.get(slice_.pid, 0.0) + (slice_.end - slice_.start)
        for process in processes:
            assert received[process.pid] == process.estimated_burst

        for prev, curr in zip(timeline, timeline[1:]):
            assert curr.start >= prev.end, "single CPU: slices must not overlap"
