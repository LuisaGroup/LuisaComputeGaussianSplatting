# LuisaComputeGaussianSplatting

## How to start

- get the ply file from Release
  - mip360_bicycle
  - nerf_blender_lego
- generate LuisaCompute Path
  - create `.env` file by copy `.env.template` file, and modify the `LC_DIR` to the path on your machine
  - `xmake l setup.lua` to generate `lc_options.generated.lua`
- build 
  - `xmake` 
- run
  - `xmake run lcgs-app --ply=<path_to_your_ply> --backend=<dx/cuda/...> --out=<dir_to_your_out_img>`
  - e.g. `xmake run lcgs-app --ply="D:/ws/data/assets/samples/gsplat.ply" --backend=cuda --output="D:/ws/data/mid/lcgs"`
  - then you can check `<dir_to_your_out_img>` with `gs_splat_<dx/cuda...>.png` for the result

## Result

### lego

![](doc/nerf_lego_result.png)

### mip360_bicycle

![](doc/mip360_bicycle_result.png)


## Known Issues

- CUDA和dx似乎在矩阵乘的计算上存在不一致，目前以cuda结果为准
- 当前只有前向计算的代码，没有反向
- 并行原语的实现比较粗暴，有提升效率的空间

