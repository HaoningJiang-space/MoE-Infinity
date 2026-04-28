from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher


class _FakeArcherEngine:
    def __init__(self):
        self.replaced = []
        self.enqueued = []

    def replace_cache_candidates(self, tensor_ids):
        self.replaced.append(list(tensor_ids))

    def get_node_default_device(self, tensor_ids):
        return 0

    def enqueue_prefetch(self, tensor_id, gpu_id):
        self.enqueued.append((tensor_id, gpu_id))


def _make_prefetcher(mode):
    prefetcher = ExpertPrefetcher.__new__(ExpertPrefetcher)
    prefetcher.archer_engine = _FakeArcherEngine()
    prefetcher.prefetch_execution_mode = mode
    prefetcher._last_prefetch_plan_candidate_count = 0
    prefetcher._prefetch_runtime_stats = {
        "prefetch_enqueue_count": 0,
        "prefetch_plan_replace_count": 0,
        "prefetch_plan_empty_replace_count": 0,
        "prefetch_plan_candidate_count": 0,
        "prefetch_plan_cleared_candidate_count": 0,
    }
    return prefetcher


def test_prefetch_execution_replace_only_skips_enqueue():
    prefetcher = _make_prefetcher("replace_only")

    prefetcher._execute_prefetch_plan([10, 11])

    assert prefetcher.archer_engine.replaced == [[10, 11]]
    assert prefetcher.archer_engine.enqueued == []
    assert prefetcher._prefetch_runtime_stats["prefetch_enqueue_count"] == 0


def test_prefetch_execution_enqueue_only_skips_replace():
    prefetcher = _make_prefetcher("enqueue_only")

    prefetcher._execute_prefetch_plan([10, 11])

    assert prefetcher.archer_engine.replaced == []
    assert prefetcher.archer_engine.enqueued == [(10, 0), (11, 0)]
    assert prefetcher._prefetch_runtime_stats["prefetch_enqueue_count"] == 2


def test_prefetch_execution_disabled_skips_both_paths():
    prefetcher = _make_prefetcher("disabled")

    prefetcher._execute_prefetch_plan([10, 11])

    assert prefetcher.archer_engine.replaced == []
    assert prefetcher.archer_engine.enqueued == []
