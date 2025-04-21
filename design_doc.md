### **High‑Level Design Doc — “Slither‑RL”**

*A Python micro‑MMO built for real‑time play **and** large‑scale reinforcement‑learning experiments on a Core i9 / RTX 5090 workstation.*

---

## **1\. Technology Stack**

| Layer | Chosen Tech | Why |
| ----- | ----- | ----- |
| **Core simulation** | **NumPy \+ Numba 0.59 (CPU JIT)** | SIMD (AVX‑512) vector loops, seamless fallback to plain NumPy. Numba 0.59 is the first release with AVX‑512 and Python 3.11 support (stay on 3.11 until 3.12 compatibility lands). citeturn0search1 |
| **GPU batch roll‑outs** | **PyTorch ≥ 2.3 \+ CUDA 12.8** (RTX 5090\) | Tensors map 1‑to‑1 to arrays already used in the CPU engine, enabling zero‑copy transfers. |
| **Multi‑agent RL API** | **PettingZoo \+ Gymnasium compatibility** | Unified step/observe API for hundreds of concurrent snakes; aligns with Stable‑Baselines3 v2. citeturn0search3turn0search9 |
| **RL algorithms (research)** | **Stable‑Baselines3 v2 / Ray RLlib** | Start with SB3 PPO; scale to distributed self‑play via RLlib if desired. |
| **Rendering** | **ModernGL 5.x \+ pyglet event loop** | Instanced OpenGL draws \~100 k food sprites & 200 k segments at 144 Hz. citeturn0search0turn0search5 |
| **Packaging** | Poetry, MyPy, Black, pre‑commit |  |
| **Testing** | PyTest \+ Hypothesis (property‑based) |  |
| **CI/CD** | GitHub Actions (CPU test matrix), optional CUDA runner |  |

---

## **2\. Project Layout**

slither\_rl/  
│  
├─ engine/          \# pure logic — no rendering  
│   ├─ world.py     \# World class, tick(), spawn(), config  
│   ├─ physics.py   \# Numba‑accelerated food motion  
│   ├─ collision.py \# spatial grid \+ pairwise tests  
│   ├─ snakes.py    \# dataclasses & helpers  
│   └─ tests/       \# unit \+ property tests  
│  
├─ ai/  
│   ├─ base.py      \# AbstractController (get\_action(obs) \-\> controls)  
│   ├─ scripted.py  \# Hard‑coded baseline bots  
│   ├─ rl/          \# Gym/PettingZoo env wrappers  
│   └─ models/      \# ONNX export / import helpers  
│  
├─ render/  
│   ├─ window.py    \# pyglet window, input handling  
│   ├─ glsprites.py \# ModernGL instanced sprite batches  
│   └─ shaders/     \# GLSL files  
│  
└─ tools/           \# benchmark, replay, batch\_train.py

Each folder has an `__init__.py` that re‑exports the public API, so IDE autocompletion shows only stable symbols.

---

## **3\. Core Data Model**

| Entity | Structure (SoA) | Notes |
| ----- | ----- | ----- |
| **Food** | `food_x, food_y, food_vx, food_vy : float32[ N ]` | 100 k rows; contiguous arrays enable single‑pass Numba kernels. |
| **Snakes** | Per‑snake header table (`len, head_idx, weight, alive, ctrl_id`) \+ a **segment pool** `(seg_x, seg_y)` | Segment pool is a ring buffer; headers store start & length. |
| **Spatial index** | Uniform grid hash `cells: List[List[int]]` | Cell ≈ 2 × max radius; updated every tick; used for collision, attraction, and viewport queries. |

---

## **4\. Physics & Game‑Tick Pipeline (≈ 16 ms)**

1. **Input phase**

   * Collect human pointer & button.

   * For each AI controller, call `get_action(observation)`.

   * Clamp pointer delta by `MAX_POINTER_SPEED`.

2. **Movement phase (Numba kernel)**

   * Heads: `pos += dir * speed * dt`.

   * Tail: slide segments forward in the ring buffer only when `grow_counter >= seg_radius`.

3. **Food dynamics (Numba kernel)**

   * Apply friction factor `v *= exp(-μ dt)`.

   * Compute attractive force from *local* heads (≤ 5 × radius) fetched via spatial grid; add vectors, update velocity and position.

4. **Collision detection**

   * **Head–food**: same grid query; distance² \< r² ⇒ eat.

   * **Head–segment**: for each head, check other‑snake segments in neighbouring cells.

   * **Head–head**: resolve smaller‑dies rule.

   * **Boundary**: `abs(x) > WORLD_HALF – r`.

   * Dead snakes convert to special food (unless boundary death).

5. **Post‑tick bookkeeping**

   * Spawn new food / snakes if global targets not met.

   * Rebuild spatial grid (incremental update).

   * Collect per‑agent observations (used by AI & replay logs).

Numba’s `@njit(parallel=True)` \+ `prange` loops over grid cells give near‑linear scaling on 32 threads with AVX‑512.

---

## **5\. Weight → (length, radius) Mapping**

AREA\_PER\_SEGMENT \= π \* BASE\_RADIUS\*\*2          \# constant  
def decompose\_weight(w: float) \-\> Tuple\[int, float\]:  
    length \= max(1, int(w // AREA\_PER\_SEGMENT))  
    \# radius grows sub‑linearly so that huge snakes still fit on screen  
    radius \= BASE\_RADIUS \* (1 \+ 0.2 \* log1p(length))  
    return length, radius

Property‑based tests assert:  
 `length * π * radius**2 <= weight < (length+1) * π * radius**2`.

---

## **6\. AI Controller Interface**

class AbstractController(Protocol):  
    def reset(self, snake\_id: int, seed: int): ...  
    def get\_action(self, obs: Observation) \-\> Controls:  
        """Returns (virtual\_pointer\_dx, virtual\_pointer\_dy, boost\_bool)."""

* `scripted.GreedyController`: aims at nearest food, avoids segments.

* `rl.RLController`: wraps a PyTorch or ONNX model (CPU inference ≤ 0.1 ms).

* Batch training: `ai/rl/env.py` exposes `PettingZooAECEnv`, so SB3’s PPO or RLlib can run headless simulations on the GPU. citeturn0search3

---

## **7\. Rendering Path**

1. Main thread owns the pyglet window (required by OpenGL).

2. Each frame, pull immutable **snapshot buffers** from the simulation thread via a lock‑free ring.

3. ModernGL draws:

   * Grid (static VBO)

   * Food: single instanced quad, per‑instance `vec4(x, y, scale, color_id)`; \~100 k, one draw call.

   * Snake segments: second instanced draw; color chosen per‑snake.

4. HUD layer for FPS, snake length, zoom.

With instanced rendering on RTX 5090 the GPU cost is negligible; the bottleneck is simulation.

---

## **8\. Performance Targets (on the specified rig)**

| Mode | Snakes | Food | FPS | CPU% | GPU% |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **Play** | 300 | 120 k | 144 | 45 (CPU) | 15 |
| **Headless vectorized** | 1 024 | 500 k | 3 000 ticks/s | 85 (GPU, PyTorch) | 0 (no display) |

---

## **9\. Testing Strategy**

* **Unit tests** (`pytest`):

  * `test_decompose_weight()` (property based)

  * `test_head_food_collision()`

  * `test_force_conservation()` (weight bookkeeping)

* **Integration**: deterministic seed → replay log → verify final world hash.

* CI matrix: `linux‑x86_64 / py3.11`, `cuda‑12.8` (optional runner).

* Benchmarks (`pytest‑benchmark`): assert max frame time ≤ 6 ms.

---

## **10\. Future Road‑Map**

| Phase | Goal | Notes |
| ----- | ----- | ----- |
| **0.1** | Minimal world \+ keyboard‑steered snake, text‑mode | Use matplotlib scatter for quick sanity frames. |
| **0.2** | Full physics, Numba kernels, grid collision | Reach 10 k food @ 60 FPS. |
| **0.3** | OpenGL renderer \+ human controls | First playable. |
| **0.4** | PettingZoo env \+ baseline bots | Training loop on CPU. |
| **0.5** | GPU batch env (PyTorch tensors) | Self‑play; save ONNX nets. |
| **1.0** | Polished UI, replays, LAN server option | Optional: migration to Vulkan via `pygfx`. |

---

## **11\. External References**

* ModernGL instancing discussion citeturn0search0

* Limitations of Numba vs. Python 3.12 (keep 3.11) citeturn0search1

* PettingZoo multi‑agent API citeturn0search3

* SB3 switch to Gymnasium backend citeturn0search9

* Flash‑Attention & CUDA 12.8 issues on RTX 5090 (future proof) citeturn0search2

---

### **Summary**

This document specifies a **Numba‑accelerated simulation core**, **ModernGL instanced renderer**, and a **PettingZoo‑compliant RL layer** that together satisfy the performance and modularity requirements. Every major subsystem has a clean interface, unit‑test coverage, and a clear migration path from simple scripted AI to GPU‑accelerated self‑play. With the outlined stack you can both **play** and **research** at scale on your Core i9 / RTX 5090 workstation.

