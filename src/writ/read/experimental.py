"""Experimental iterators for reading and transforming.

Currently, this contains an implementation for the threaded evaluation of iterables.
This has perceived utility when dealing with i/o intensive generators, although the
efficiency gain is not consistent.

The content of this module is experimental and should be used with caution.
In particular, you should be aware of the thread-safe properties of the underlying 
libraries when threaded approaches are used.
"""

from typing import Iterable, Iterator, Any, TypeVar, Generic
import threading
import queue
import time

A = TypeVar("A")


class SentinelThreader(Generic[A]):
    """Iterates over a target in a separate thread and serves results.

    This class yields the same values a wrapped source iterable does, but it
    evaluates the wrapped iterable in a new thread and caches the value. This
    may speed up i/o intensive generators when they are part of a
    compute-intense pipeline.

    Warning:
    -------
    This is an extremely bare-bones implementation of evaluating an input
    generator in a separate thread. It communicates the end of an iterable
    via a sentinel value, which defaults to None. If your iterator returns
    this sentinel, all bets are off.

    We don't clean up the thread on deletion, but it should not consume
    any memory or be doing anything.

    This is experimental, use with caution. We only use a single thread,
    so it seems unlikely we will get into hairy thread behavior.

    Important Attributes:
    --------------------
    source:
        Source iterable for the thread to draw from.
    sentinel:
        The value which is used by the thread to mark the end of the iteration.
        Cannot be normally served by source.
    """

    def __init__(
        self,
        source: Iterable[A],
        cache_size: int = 1,
        sentinel: Any = None,
        defer: bool = True,
    ) -> None:
        """Initialize threader.

        Arguments:
        ---------
        source:
            Iterable the thread will pull values from.
        cache_size:
            Roughly the number of values to pull values ahead of current location. If
            this is to low, performance can go down, but larger values require more
            memory.

            Note that cache_size+1 or cache_size+2 values may actually be held in memory
            at once-- a value of 1 will still prefetch, and may be good for many i/o
            applications.
        sentinel:
            Object to use as sentinel. source cannot return this object during
            iteration as doing so would cause premature termination. If it does,
            we raise an ValueError in the thread, although this probably does not
            get propagated back to the main thread.
        defer:
            If truthy, we
        """
        self.cache_size = cache_size
        assert iter(source)
        self.source = source
        self.sentinel = sentinel
        self.defer = defer

        # this should not go above this.
        self._input: queue.Queue = queue.Queue(maxsize=2)
        self._thread = self._start()
        self._active = False

    def __iter__(self) -> Iterator[A]:
        """Start threaded iteration.

        One thread exists continuously for all iterations.

        If iteration is interrupted via ^C, you likely will need to create a new
        object for the same functionality. Threads may not be stopped easily
        in an interactive session, but should die in a script as expected due
        to daemon options.
        """
        # this means the iterator was entered and has not yet exited properly.
        if self._active:
            raise ValueError(
                "Threaded reader is not reeentrant and our state "
                "is dirty. This may be also be due to a partial execution. "
                "due to interruption."
            )
        else:
            self._active = True

        # we put a queue in the input queue to signal for the thread to start.
        q: queue.Queue = queue.Queue(maxsize=self.cache_size)
        self._input.put(q)

        # now wait for thread output.
        while True:
            # attempt read, check for sentinel.
            g = q.get()
            # defer thread control if possible.
            time.sleep(0)
            if g is self.sentinel:
                break
            try:
                yield g
            except GeneratorExit:
                break
        # tell child thread to drop its pending value when it checks for next object
        self._input.put(None)
        # trigger next object get to make sure worker sees the pushed None
        try:
            _ = q.get(block=False)
        except queue.Empty:
            pass
        self._active = False

    def _start(self) -> threading.Thread:
        """Start thread worker."""

        def worker() -> None:
            """Take queue entries from input queue and fill them with results.

            There are two roles of queues in this function: self._input is a
            queue that itself holds queues. Each of these member queues
            represent an iteration task.  We wait for a queue to show up in
            self._input and then start filling it with values.

            Control logic gets messy, as if iteration gets stopped early, we must tell
            this worker to reset. This is done by passing a None into self._input
            externally. In order to avoid stalling, we must use timeouts and loops.
            """
            # outer loop looks for queues in self._input. If we find a non-None
            # entry, we start pulling from source to fill it.
            while True:
                current_queue = self._input.get()
                # a None value can be pushed as a null input to kill a previous
                # iteration.
                if current_queue is None:
                    continue
                for x in self.source:
                    # check to see if we get a sentinel value from our iterator.
                    # we shouldn't--- we use the sentinel to signal an iterator
                    # is done. If we find one, quit.
                    if x is self.sentinel:
                        raise ValueError(
                            "Encountered sentinel {self.sentinel} in "
                            "iteration. Thread execution is bad, question "
                            "results."
                        )
                    # we now know we have a real value to return.
                    # repeatedly attempt to place item in output queue.
                    current_queue.put(x, block=True)
                    # immediately remove our reference to allow garbage collection
                    del x
                    if self._input.empty() is False:
                        break

                # if we run out of elements in the iterable, return sentinel.
                else:
                    current_queue.put(self.sentinel)

        # Turn on the worker thread.
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread
