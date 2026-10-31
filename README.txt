MOFSynth Fast 3.1.1 — COMBINED EHULL CONVERGENCE (2026-10-24)

Based on mofsynth-fast-3.1-linkerfix-v2-samax.zip. All linker parent,
SA/HOMO and SA-max-target RMSD/DeltaE changes from that package are retained.
The default Ehull route now requires atomic force <0.01 eV/A, maximum absolute
principal stress <0.05 GPa at zero pressure, filtered generalized force <0.03,
and E/N span <=1e-5 eV/atom over the last five updates (six configurations),
with all gates satisfied for five consecutive optimizer steps. Max steps: 10000.
See EHULL_COMBINED_3_1_1.txt for settings, receipts, validation and limits.

Use fresh --work and batch --out directories for this release. Old force-only
Ehull receipts and cached predictions are rejected by the default route.
--no-fast-ehull disables hull overlap only, retaining combined convergence.
--ehull-solver orig changes the BFGS solver, retaining combined convergence.
The frozen --baseline route remains the explicitly historical implementation.
Restart and random-perturbation checks are validation diagnostics and are not
performed automatically by the default inference path. No lowest-of-repeats
selection, model retraining or recalibration is performed.

The lightweight package reuses an existing runtime; no environment reinstall is
needed. Pass --prefix EXISTING_RUNTIME --ehull-python EXISTING_GPU_PYTHON to
predict.sh, or configure this package with install.sh --reuse as described below.
The installed/shared runtime is not patched: combined convergence is package-local.
Historical validation/performance files below remain evidence of their original
protocols. The original 3.1 speed/equivalence claims are not acceptance claims for
3.1.1, and the previously quoted numerical ranges are not universal error bounds.

RMSD / DeltaE selection revision v4, 2026-10-25
Protocol: rmsd-deltae-sa-stereo-compatible-v4-20261025
1. Prefer exact standardized-parent SMILES equality with the complete-set SA maximum.
2. Only if no exact match: for a wholly stereo-unspecified target, compare parents
   after removing stereochemistry ONLY. Preserve isotope labels, atom/bond graph,
   bond orders, formal charge, H count and tautomer; no chemical-state edits.
   Targets with ANY specified stereo, including partial specification, do not use
   this tier. Correspondence does not prove the same absolute stereoisomer.
3. If neither matches, use original Open Babel StripSalts (-r), including native ties.
Within either matching tier, choose most explicit atoms, then first component.
Keep target/selected parent strings and original explicit atoms/coordinates intact.
Receipts distinguish exact_sa_parent / stereo_compatible_sa_parent /
largest_fragment_fallback. selected_matches_sa_parent and n_matching_components
retain their exact-match meaning; selected_corresponds_sa_parent includes exact
or compatible correspondence. stereo_compatibility_used and
n_stereo_compatible_components report the second tier explicitly.
New protocol rejects all old protocol caches; use fresh work directories.
Complete finite SA set still required. SA/HOMO/Ehull and xTB settings unchanged.
No batch restart or historical feature replacement. This revision is included
in the updated ZIP; see REPACK_20261025.txt.
Supersedes selection rules in RMSD_SA_FALLBACK_V3.txt and RMSD_SA_MAX_V2.txt.

LINKER POSTPROCESSING FIX v1 (2026-10-24)

The default predict/ pipeline now uses a versioned neutral-parent and canonical-tautomer rule.
This intentionally changes SA/HOMO for affected fragments. The original 3.1 archive and the
baseline/ implementation remain frozen. The 3.1 equivalence/performance claims below describe
the original accelerator, not validation of the corrected feature definition against a model.
See LINKER_POSTPROCESS_FIX.txt, validation/linkerfix/, and linker-postprocess-fix.patch.

Use the original RAW MOFid linker_stats.txt, not previously hydrogen-capped SMILES.
Install/reuse the runtime as described below. The source changes are already applied here.
Run validation/test_linker_postprocess.py with the runtime Python to check the patch.
No model retraining, population reprocessing, or calibrated score validation is included.

HISTORICAL 3.1 DOCUMENTATION — settings and equivalence claims below are superseded where stated above

MOFSynth Fast 3.1 — 五特征 MOF 可合成性评分（加速版）

两个 ZIP 二选一。完整包包含轻量包的全部内容，不要在同一目录重复叠加解压。

完整离线包：解压到一个可写的新目录，然后运行：
  bash install.sh --plan
  bash install.sh
  bash predict.sh --cif examples/reference58.cif --label demo --gpu-id 0

安装不需要网络或 root，不修改其他安装、不覆盖共享 /tmp/mofid-build。
Python、xTB、MOFid、Java、MOFSimplify、Torch/Fairchem、模型和固定凸包均随完整包提供。
只支持 Linux x86_64。已验证：RTX 4090（Ubuntu 22.04）、RTX 5090（WSL2 Ubuntu 26.04）、A800（Ubuntu 22.04）。
本地原生程序需要 glibc >= 2.34、GLIBCXX >= 3.4.29，内核允许非特权用户命名空间。
系统需有 Python >= 3.10、bash、tar、timeout、sed、grep；bubblewrap 有包内副本作为备用。
完整包连同解压、安装约需 40 GB 可用空间。--device cpu 可显式选择 CPU 放宽，耗时会增加。

轻量包：复用已经安装的 portable runtime prefix 和兼容的 Ehull Python：
  bash install.sh --reuse --prefix /absolute/path/runtime --ehull-python /absolute/path/gpu-env/bin/python
  bash predict.sh --cif /absolute/path/mof.cif --gpu-id 0
--reuse 只写本包 runtime.json，不修改现有 prefix。完整包也支持此模式。

单个 MOF：输出默认放在 runs/<label>/prediction.json，也可用 --work 指定绝对路径。
已有工作目录会沿用原规则复用中间结果；换 CIF 或公平重测应使用全新的 --work 目录。
常用选项：--gpu-id、--cpus 8、--cpu-list、--workers 2、--stage-jobs 3、--out。
bash predict.sh --help 查看入口选项；--plan 检查实际模型和环境路径。
bash predict.sh --baseline ... 调用随包冻结的原始串行实现，作为对照。

批量（推荐用于多个 MOF）：
  python3 predict_batch.py --cifs a.cif b.cif ... --out DIR --slots 5 --cpus-per-slot 6 --gpus 0
  python3 predict_batch.py --cif-list list.txt --out DIR --slots 8 --cpus-per-slot 8 --gpus 0,1
每个 CIF 是一次独立的 predict.sh（同一代码路径），多个槽位并发、各占互不重叠的 CPU 段，GPU 轮流分配。
单个结构放宽只用到 GPU 的一小部分，多槽位共享一张卡可显著提高吞吐。结果汇总在 DIR/summary.csv。
自行并发多个 predict.sh 时请用 --cpu-list 分开 CPU；不指定时每个实例都绑定最后 8 个逻辑 CPU。

安装自检：
  python3 verify.py --quick
  python3 verify.py --gpu-id 0
--quick 只检查路径与 CUDA 环境；不带 --quick 将用全新目录计算包内参考 MOF 并核对五特征与分数。
原始流程的 RMSD（xTB 气相优化）随 CPU 指令集略有差异（同一原始实现：EPYC 0.83469，Ryzen 0.83294），
自检对 RMSD 采用 5e-3 Å 窗口、分数 5e-4，并在 info 中报告是否与 EPYC 参考值逐位相同。

加速内容（均不改变结果：与原始实现逐位相同，或调用同一函数）：
  • 周期距离矩阵：保持 27 个周期镜像和原浮点操作次序的批量内核。
  • RMSD 的 2x 超胞 MOFid：只做 sbu 片段分解（该支路只读取 MetalOxo/linkers.cif，不读拓扑），并使用只替换了
    ring.cpp 的 libopenbabel：SSSR 环去重与冗余判定改为哈希和逐原子计数，算法、环列表及顺序与原始库相同，
    大超胞上 sbu 从数百秒降到数秒。原胞 MOFid 使用原始库与原始调用。运行时 libopenbabel 与补丁所基于的
    构建哈希不一致时自动回退原始调用。源码与补丁见 acceleration/openbabel/。
  • 阶段并行：Ehull、RMSD、MOFSimplify 与原胞 MOFid 同时启动；channel-B 在片段集合就绪后启动。
    每个阶段是独立进程，失败或收到 SIGTERM 时清理全部子进程。
  • Ehull：固定凸包在放宽进行时由子进程预先加载并缓存，放宽结束后调用同一个 get_energy_above_hull；按需导入。
  • BFGS 步长：Hessian 正定时用 Cholesky 求 -H^-1 g（此时与 ASE 的 V|w|^-1 V^T g 数学相同），否则用原 eigh。
    两者步长相对差约 1e-12，比原始 GPU 模型每次调用自带的力噪声（约 3e-6 eV/Å，相对约 5e-7）小 5–6 个数量级；
    大结构每步省去一次稠密特征分解（436 原子约 150 ms）。
  • 隔离环境保留 CUDA 驱动的 JIT 缓存（~/.nv/ComputeCache）。部分 kernel 以 PTX 形式提供的 GPU（如 RTX 5090）
    否则每次运行都会重新编译，放宽每步耗时翻倍以上；缓存只保存同一编译结果。
  • Zeo++ 相同源代码严格浮点优化编译，没有 fast-math、FMA 收缩或采样数变更。
  • RAC 不变量在函数内复用；RMSD 的四种 calculate_rmsd 方法在一个解释器中执行，保留全部结果标签。
  • 限制每个进程的 BLAS/OpenMP/TensorFlow 线程与 CPU 亲和性。
没有更换模型、训练权重、精度、probe 参数、MC 采样数、xTB 协议或放宽阈值。
可单独关闭：--no-fast-pbc、--no-fast-rmsd、--no-fast-hull、--no-fast-sbu、--no-early-stages、--no-fast-ehull，
--ehull-solver orig 恢复原 eigh 步长。

验收（详见 validation/acceptance/RESULTS.txt 与 evidence/）：
  • CPU 四特征及全部中间量（片段集合、逐片段 HOMO、SA、RMSD 各方法值、192 列描述符）与同机原始实现逐位相同。
  • Ehull 与分数的差异在原始实现自身重复运行的波动范围内；放宽全部收敛。
  • 5 个验证 MOF 的完整流程，与原始实现相比：RTX 5090 3.43 倍，RTX 4090 2.44 倍，A800 2.85 倍（单个 MOF 最高约 10 倍）；
    5 个 MOF 并发批量的吞吐为原始实现顺序运行的约 7.6 倍（5090）与 6.8 倍（A800）。详见验收结果中的平台表与判定。

原始流程自身的不确定性（本包未改变，解读结果时请注意）：
  • MOFid 原胞分解输出的片段 SMILES 依赖内存地址：sbu 的 AtomSet 是按指针排序的 std::set<OBAtom*>，导出片段时的
    原子顺序随堆布局（受工作目录路径长度影响）变化，进而改变键级/自由基位置。包内 gen_eed3ec90c711abe1（四唑配体）
    已出现 4 种写法，标准化后对应不同的四唑互变异构体，Linker HOMO 为 -10.598/-10.359/-10.235/-9.958 eV，
    分数在约 0.23–0.40 之间。另外 4 个验证 MOF 未出现。路径长度相同时本包与原始实现逐位相同。
  • GPU 放宽不逐次复现：同一构型重复调用模型，力相差约 3e-6 eV/Å。原始实现在 4090/5090/A800 上的 104 次放宽
    （不含参考 MOF）中有 10 次落到另一个能量极小（Ehull 偏离约 1.5e-4–8.3e-4 eV/atom，分数偏离可达约 3e-3）。

默认关闭的开关：--ehull-inference nockpt_merge（关闭激活重算并合并 MoLE 专家，fp32 不变，模型调用约快 15–30%）。
它对力的扰动与原始实现自身噪声同量级；落入替代极小的频率：单独使用 4/21，与 Cholesky 合用 9/35，原始实现 10/104
（同卡同批对照 6/24 对 4/24），证据不足以认定等价，因此不默认启用。

最终分数保持原项目的含义，不代表实验可合成性的验证结论。
源代码、严格编译命令及第三方授权：sources/、LICENSES/、acceleration/openbabel/（OpenBabel，GPL-2.0）。
完整包 payload/base-bundle.tar.gz 保留原始上游分发包及其全部授权和源文件。
