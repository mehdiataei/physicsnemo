# PhysicsNeMo Geometry Examples

The standalone `physicsnemo.geometry` adapters accept
`physicsnemo.mesh.Mesh` and, where documented, `DomainMesh` objects. The
examples in this directory use `Mesh`. Tensor-only kernels remain available
from `physicsnemo.nn.functional`.

From the repository root, install PhysicsNeMo and run the example:

```bash
pip install -e .
python examples/minimal/geometry/deformation_energy_optimization.py
```

## Differentiable Deformation Energy Optimization

Run `deformation_energy_optimization.py` for a compact shape-optimization
example. It preserves a prescribed radial-basis handle displacement while
penalizing strain, total-area change, and element inversion. The script uses
Warp on CUDA when available and falls back to Torch on CPU.

## Sobolev Shape Optimization

Run `sobolev_shape_optimization.py` to compare direct dense displacement with
P1 Sobolev deformation on a small triangulated square. The script optimizes
candidate vertex coordinates against a noisy target and checks that both
objectives decrease, the Sobolev adjoint varies more smoothly between
neighboring vertices, and fixed boundary vertices do not move.

The example selects CUDA when available and otherwise runs on CPU. CUDA
segments, triangles, and tetrahedra use the Warp backend by default. CPU
meshes use Torch. The example has no plotting dependency. The reproducible
figure source is `docs/img/geometry/sobolev_adjoint_field.py`.

## 3D Sobolev Sheet Shape Optimization

Run `sobolev_surface_shape_optimization.py` to apply the same optimization to
a triangulated sheet embedded in three dimensions. The objective pulls its
center upward while the boundary remains fixed. The example checks loss
reduction, fixed anchors, and smoother vertex adjoints. The reproducible figure
source is `docs/img/geometry/sobolev_adjoint_field_3d.py`.
