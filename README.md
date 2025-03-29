# LuisaComputeGaussianSplatting

## How to start

- generate LuisaCompute Path
  - create `.env` file by copy `.env.template` file, and modify the `LC_DIR` to the path on your machine
  - `xmake l setup.lua` to generate `lc_options.generated.lua`
- build 
  - `xmake` 
- run
  - `xmake run lcgs-app --ply=<path_to_your_ply> --backend=<dx/cuda/...>`
  - e.g. `xmake run lcgs-app --ply="D:/ws/data/assets/samples/gsplat.ply" --backend=cuda`

## Know Issue

- CUDA和dx似乎在矩阵乘的计算上存在不一致
- 浮点精度损失有点大，考虑和原算法保持一致
