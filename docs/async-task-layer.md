# Design: async task layer for `aoti-rs`

Status: **proposal, not yet implemented.** This document exists to be argued
with before any code is written.

## 1. Goal

Let a service accept concurrent inference requests, hand them to `aoti-rs`, and
await results — while giving the service author enough visibility into the
pending work to write a *scheduling policy* suited to their workload
(latency-sensitive interactive traffic, throughput-oriented bulk jobs, mixed
priorities, deadlines).

The premise is that the interesting part is not "make `run()` awaitable" — that
is a channel and twenty lines. The interesting part is **exposing task metadata
in a form that makes non-trivial schedulers writable by a user of the crate**,
without handing them the tensors or the model.

### Non-goals (v1)

- Dynamic batching / request coalescing. Designed for in §10, deliberately out
  of v1's scope.
- Distributed scheduling, multi-process, or multi-host work distribution.
- Model loading/unloading at runtime (LRU model cache, swapping weights).
- Anything resembling a full serving framework: no HTTP, no gRPC, no metrics
  exporter. The layer emits data; the user wires it to their stack.

## 2. Constraints imposed by `aoti-rs` today

These are read off the current code, and they shape everything below.

| Fact | Where | Consequence |
| --- | --- | --- |
| `AOTIModel<D>` is `Send`, not `Sync` | `src/lib.rs:684` | A model can be moved to a worker thread, but not shared between them. |
| `run`/`boxed_run` take `&mut self` | `src/lib.rs:713`, `src/lib.rs:725` | Calls serialize per model. **`num_runners > 1` currently buys nothing** — the C++ pool exists (`csrc/aoti.cc:48`) but `&mut self` prevents concurrent entry. |
| `boxed_run` takes `Vec<DeviceTensor<D>>` by value | `src/lib.rs:725` | A task queue owns its inputs, so `boxed_run` is the natural call — the in-place buffer-reuse optimization comes for free. |
| `get_call_spec` / `get_constant_fqns` take `&mut self` | `src/lib.rs:745`, `src/lib.rs:750` | A scheduler must never call these; they'd contend with inference. Snapshot at load time instead. |
| Device kind is a type parameter, checked once at load | `src/lib.rs:598` | The task layer should stay generic over `D` and preserve the guarantee, not erase it. |
| Inference is blocking, CPU/GPU-bound | — | Compute must never touch an async executor thread. The async surface is submission/completion only. |

### 2.1 Verified upstream behavior

The concurrency design below depends on PyTorch internals, so they were checked
against the source rather than assumed:

- `AOTInductorModelContainer` keeps `available_models_` and `pending_models_`,
  guarded by `models_mutex_` with a `pending_models_available_` condition
  variable. `run()` takes `model_exec_mutex_` **in shared mode** ("we acquire
  the mutex in shared mode if we allow concurrent execution"), checks out a
  model instance, and returns it to the pending queue on completion.
  **`run()` is safe to call concurrently from multiple threads.**
- With `num_models == 1`, concurrent callers are correct but fully serialized —
  each waits for the single instance to come back. Concurrency requires
  `num_runners > 1`.
- `AOTIModelContainerRunner::run` and `boxed_run` are non-`const` methods, but
  they do not mutate any member of the runner: they read `run_func_`,
  `container_handle_`, `get_num_outputs_func_`, `proxy_executor_handle_`,
  `get_last_error_func_` and delegate. The non-`const` marking is incidental.
- **`run_single_threaded` is the trap.** It bypasses the locks entirely and
  indexes `available_models_[0]` directly; upstream describes it as
  "obviously unsafe to use in a threaded environment". Our builder exposes this
  as `.single_threaded(bool)` (`src/lib.rs:578`), so the shared-model API must
  refuse to coexist with it.

Because this rests on upstream implementation details, it is pinned to the
PyTorch version behind `tch = "=0.24.0"` and must be re-verified on any libtorch
bump. That re-check belongs in the upgrade checklist, not in a comment nobody
reads.

## 3. Architecture

```
  async caller                 engine (owns payloads)              worker threads
 ────────────────    ──────────────────────────────────────    ────────────────────
  submit(task) ──────▶ admission ──▶ pending queue
                                        │
                                        │  Scheduler::pick(&[TaskMeta]) -> TaskId
                                        ▼
                                     dispatch ─────────────────▶ SharedModel<D>.run(&self)
                                                                        │
  await TaskHandle ◀──── oneshot ◀──── completion ◀─────────────────────┘
                                        │
                                        └─▶ Scheduler::on_complete(meta, outcome)
```

Three properties are load-bearing:

1. **The engine owns the payloads; the scheduler sees only metadata.** A
   scheduler receives `&[TaskMeta]` and returns a `TaskId`. It never holds a
   `DeviceTensor`, never deals with cancellation bookkeeping, and cannot leak a
   tensor onto the wrong thread. This is what makes user-written schedulers
   both safe and easy.
2. **Compute is on dedicated OS threads.** The engine spawns them itself; it
   does not require or assume an async runtime.
3. **Device typing survives.** `Engine<D>`, `Task<D>`, `TaskHandle<D>`. A CPU
   engine and a CUDA engine are separate objects.

## 4. Core types

### 4.1 Static metadata — snapshotted once, shared freely

```rust
pub struct ModelInfo {
    pub id: ModelId,
    pub name: String,              // model name within the package
    pub package_path: PathBuf,
    pub device_key: String,        // AOTI_DEVICE_KEY as found in the package
    pub call_spec: Vec<String>,    // captured at load; needs &mut on the model
    pub constant_fqns: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub num_runners: usize,        // effective inference parallelism
    pub input_arity: Option<usize>,
    pub output_arity: Option<usize>,
}
```

Held as `Arc<ModelInfo>` and handed to the scheduler. Everything here is read
from the model exactly once, at registration, so no scheduler decision ever
contends with an in-flight `run()`.

### 4.2 Per-task metadata — the scheduler's input

```rust
pub struct TaskMeta<T = ()> {
    pub id: TaskId,
    pub model: Arc<ModelInfo>,
    pub enqueued_at: Instant,
    pub deadline: Option<Instant>,
    pub priority: u8,
    pub inputs: SmallVec<[TensorDesc; 4]>,  // shape, dtype, numel, bytes
    pub tag: T,                             // user-defined, e.g. tenant id
}

impl<T> TaskMeta<T> {
    pub fn waited(&self) -> Duration;        // now - enqueued_at
    pub fn slack(&self) -> Option<Duration>; // deadline - now - est_cost
    pub fn batch_size(&self) -> Option<i64>; // dim 0 of first input
    pub fn input_bytes(&self) -> usize;
    pub fn is_canceled(&self) -> bool;       // handle dropped
}
```

`TensorDesc` is derived at submit time from the `DeviceTensor` (`size()`,
`kind()`, `numel()`) — cheap, no device sync, no copy.

The generic `tag: T` is what makes this usable for real policies: fair-share
across tenants, per-customer quotas, request-class routing. The crate should not
try to enumerate those concepts; it should carry the user's.

### 4.3 Completion

```rust
pub struct Outcome {
    pub queued: Duration,     // enqueued -> dispatched
    pub ran: Duration,        // dispatched -> returned (see §8 re: CUDA)
    pub result: Result<(), ErrorKind>,
    pub queue_depth_at_completion: usize,
}
```

Fed back into `Scheduler::on_complete` and into the cost model.

## 5. The `Scheduler` trait

```rust
pub trait Scheduler<D: Device, T = ()>: Send + 'static {
    /// Admission control. Runs on the submitting thread, before enqueue.
    /// Reject here to shed load (e.g. deadline is already unreachable).
    fn admit(&mut self, task: &TaskMeta<T>, ctx: &SchedCtx) -> Admit {
        Admit::Accept
    }

    /// Choose the next task to dispatch. Called whenever a worker is free
    /// and the queue is non-empty. Returning `None` parks the worker until
    /// the next enqueue or the returned deadline elapses.
    fn pick(&mut self, pending: &[TaskMeta<T>], ctx: &SchedCtx) -> Pick;

    /// Feedback. Called after each task finishes.
    fn on_complete(&mut self, task: &TaskMeta<T>, outcome: &Outcome) {}
}

pub enum Admit { Accept, Reject(RejectReason) }

pub enum Pick {
    Task(TaskId),
    /// Deliberately idle — e.g. a batching policy waiting for more arrivals.
    /// The engine re-polls after `wake_in`, or sooner on a new enqueue.
    Idle { wake_in: Option<Duration> },
}
```

`SchedCtx` gives the scheduler the state it can't derive from the task list
alone:

```rust
impl SchedCtx {
    pub fn now(&self) -> Instant;
    pub fn in_flight(&self) -> usize;
    pub fn idle_workers(&self) -> usize;
    /// EWMA of observed runtime for this model at this batch size. See §8.
    pub fn est_cost(&self, meta: &TaskMeta) -> Option<Duration>;
}
```

`pick` returning an index rather than draining the queue is a deliberate
restriction: it keeps the engine in control of ordering, cancellation, and
lifetime, and it means a buggy user scheduler can starve a task but cannot
corrupt state or drop a caller's oneshot on the floor. Starvation is
observable — `TaskMeta::waited()` is right there — and the built-ins avoid it.

### Built-in schedulers

- `Fifo` — the default; also the reference for "did my scheduler help?"
- `Priority { aging: Option<Duration> }` — strict priority, optionally aging
  waiters up to avoid starvation.
- `Edf` — earliest deadline first, with optional drop-if-unreachable using
  `est_cost`.
- `ShortestFirst` — `est_cost` ascending; good throughput, bad tail latency.
  Included mostly to make the trade-off visible.

## 6. Concurrency: `SharedModel<D>`

The chosen topology is one model instance shared by `num_runners` workers,
rather than N model instances. On CUDA the alternative means N copies of the
weights in VRAM, which is usually prohibitive.

This requires a change to the **core** crate:

```rust
/// A model whose C++ runner is entered concurrently by up to `num_runners`
/// threads.
pub struct SharedModel<D: Device> { /* wraps AOTIModel<D> */ }

impl<D: Device> SharedModel<D> {
    pub fn run(&self, inputs: &[DeviceTensor<D>]) -> Result<Vec<DeviceTensor<D>>, Error>;
    pub fn boxed_run(&self, inputs: Vec<DeviceTensor<D>>) -> Result<Vec<DeviceTensor<D>>, Error>;
}

unsafe impl<D: Device> Sync for SharedModel<D> {}
```

Construction is fallible and enforces the two preconditions from §2.1:

```rust
impl<D: Device> AOTIModel<D> {
    /// Errors if the model was built with `single_threaded(true)`, which
    /// selects a container path that is explicitly not thread-safe.
    /// Warns (or errors) if `num_runners == 1`, where sharing is correct
    /// but yields no parallelism.
    pub fn into_shared(self) -> Result<SharedModel<D>, Error>;
}
```

### 6.1 Keeping the `unsafe` in one place

cxx requires `Pin<&mut T>` for non-`const` C++ methods, which is why the bridge
looks the way it does today (`src/lib.rs:78`). Rather than casting away
constness on the Rust side, add const-taking overloads in `csrc/aoti.h`:

```cpp
rust::Vec<OwnedTensor> runner_run_shared(
    const torch::inductor::AOTIModelContainerRunner& runner,
    const rust::Vec<TensorPtr>& inputs);
```

with a single documented `const_cast` inside the C++ implementation. cxx maps a
`const T&` parameter to `&T` in Rust, so `SharedModel::run(&self)` falls out
naturally and **Rust performs no pointer manipulation at all**. The soundness
argument (§2.1: runner methods mutate no runner member; the container serializes
its own pool) then lives next to the `const_cast` that relies on it.

This is the one genuinely unsafe thing in the design, and it is worth being
loud about: it is justified by upstream implementation details, not by upstream
API contract.

## 7. Async surface and cancellation

The core is executor-agnostic. `TaskHandle<D>` is a `Future` over a
`futures-channel::oneshot::Receiver` — no tokio dependency in the base crate.

```rust
let handle = engine.submit(Task::new(model_id, inputs).deadline(t).priority(3))?;
let outputs: Vec<DeviceTensor<Cpu>> = handle.await?;
```

- `handle.blocking_recv()` for synchronous callers.
- A `tokio` feature adds only conveniences (spawning the engine on a runtime,
  `tokio::sync` interop). It is not required.

**Cancellation is cooperative and honest about its limits.** Dropping a
`TaskHandle` closes the oneshot; the engine checks `is_canceled()` at dispatch
and skips the task. A task already inside `run()` cannot be canceled — AOTI
offers no interruption mechanism, and we should say so in the docs rather than
implying otherwise. Schedulers can see cancellation via
`TaskMeta::is_canceled()` and deprioritize accordingly.

## 8. The cost model, and why CUDA makes it subtle

`SchedCtx::est_cost` is an EWMA of observed runtime keyed by
`(ModelId, batch-size bucket)`. Cheap to maintain, and it's what separates a
working EDF scheduler from a guess.

**On CUDA, the naive measurement is wrong.** `run()` returns once kernels are
*enqueued* on the stream, not once they complete. Without a sync, every run
looks fast, the cost model is fiction, deadline scheduling silently degrades,
and — worse — the completion future resolves before the output tensors are
actually ready. A caller that then copies to host blocks at the copy instead.

So the CUDA engine synchronizes after each run by default
(`EngineConfig::sync_after_run`, defaulting to `true` on CUDA, `false` on CPU
where it is meaningless). Users who know their consumer stays on-device and
handles stream ordering themselves can turn it off, and accept that `Outcome.ran`
and `est_cost` become enqueue times rather than completion times.

This is worth calling out in the top-level docs. It is the kind of detail that
makes a benchmark look great and production look mysterious.

## 9. Backpressure

Two independent mechanisms, because they answer different questions:

- **Bounded queue** (`EngineConfig::max_pending`): `submit` returns
  `Err(Full)`. This is a resource bound — it protects memory, since every
  pending task pins its input tensors.
- **`Scheduler::admit`**: a policy decision — "this request cannot meet its
  deadline, fail it now rather than burn a GPU slot on it." Runs on the
  submitting thread, so it must be cheap.

## 10. Deferred: dynamic batching

Not in v1, but the design should not preclude it. Sketch:

- Per-model opt-in at registration: `.batch_dim(0)`. Only valid for packages
  exported with a dynamic batch dimension — the crate cannot verify this from
  metadata, so it is the user's assertion.
- The engine computes an eligibility key per task: `(model, dtype,
  non-batch shape)`. Only tasks with equal keys may merge.
- `Pick` grows a `Batch(SmallVec<[TaskId; 8]>)` variant. The engine validates
  the proposed batch against the eligibility keys and falls back to sequential
  dispatch if a scheduler proposes an illegal one — a user policy bug should
  degrade throughput, never correctness.
- Merge is `Tensor::cat(dim)`, split is `narrow` per output. Both are real
  copies; the win has to exceed them, which is why `BatchWindow { max_batch,
  max_wait }` needs `Idle { wake_in }` to trade latency for occupancy.

Deferring this is a judgment call worth revisiting: for request-serving on a
GPU, batching is usually where the throughput actually comes from, and the
`Pick::Idle { wake_in }` hook exists mainly to keep the door open.

## 11. Crate layout

A cargo workspace:

- `aoti-rs` — unchanged in spirit. Gains `SharedModel<D>` and the const-taking
  bridge functions. No futures, no channels, no async.
- `aoti-serve` — engine, tasks, schedulers. Depends on `aoti-rs`.

Rationale: the core stays buildable and reviewable in sync/embedded contexts,
keeps its dependency surface small, and moves on its own semver track. A
`features = ["async"]` flag on one crate would drag futures into every
consumer's lockfile and make the feature-gated public API awkward to document.

## 12. Open questions

1. **Multi-model engines.** v1 assumes one `Engine<D>` per device kind, holding
   possibly several models. Should a scheduler be able to see across models
   (e.g. to prefer whichever model has a warm cache)? That argues for one engine
   holding all models of a device kind, which is the current sketch — but the
   worker-to-model assignment policy then becomes a scheduler concern too, and
   `Pick` may need to name a worker as well as a task.
2. **Does `num_runners` parallelism actually pay on CPU?** Each AOTI model
   instance runs multithreaded internally via ATen's thread pool. Sharing one
   process's cores across N concurrent runs may just cause contention. Wants
   measurement before the default is anything but `num_runners = 1` on CPU.
3. **Priority of `into_shared` erroring vs. warning on `num_runners == 1`.**
   Erroring is honest; warning is friendlier for the common single-runner case
   where the user just wants async submission without concurrency.
4. **Should `Engine` own model loading?** Loading extracts the `.pt2` to a temp
   dir (`src/lib.rs:592`) and is slow. Registration taking an already-built
   `AOTIModel<D>` keeps the engine simple and testable; a convenience
   `Engine::load()` can wrap it.

## 13. Suggested milestones

1. `SharedModel<D>` + const bridge overloads in `aoti-rs`, with a stress test
   that hammers one shared model from N threads under ASAN/TSAN. **This is the
   riskiest piece and should be proven before anything is built on it.**
2. Engine + worker pool + `Fifo`, oneshot completion, cancellation-at-dispatch,
   bounded queue. No scheduler trait exposed publicly yet.
3. Expose `Scheduler`, `TaskMeta`, `SchedCtx`; add `Priority` and `Edf`; add the
   EWMA cost model and the CUDA sync policy.
4. Benchmarks comparing `Fifo` vs `Edf` vs `ShortestFirst` on a mixed workload —
   the deliverable that shows whether any of this was worth it.
5. (Separately decided) dynamic batching per §10.
