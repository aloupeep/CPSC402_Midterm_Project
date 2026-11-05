# E0 teaching-case replay (cb384647987dbe4e, n=64)

| record | status | iters | eta (numpy FP64) | eta (mpmath 60d) | omega | forward vs xref | forward<=10u32 |
|---|---|---|---|---|---|---|---|
| core/cb384647987dbe4e__direct16 | direct | 0 | 2.343e-04 | 2.343e-04 | 1.417e-03 | 1.0773e-01 | False |
| core/cb384647987dbe4e__direct32 | direct | 0 | 2.713e-08 | 2.713e-08 | 1.627e-07 | 1.0527e-05 | False |
| core/cb384647987dbe4e__direct64 | direct | 0 | 5.562e-17 | 5.526e-17 | 4.561e-16 | 3.3419e-14 | True |
| core/cb384647987dbe4e__half_factors_solve32 | unchanged_working_update | 9 | 1.763e-09 | 1.763e-09 | 6.958e-09 | 3.0261e-08 | True |
| core/cb384647987dbe4e__hdd | iteration_limit | 30 | 4.449e-17 | 3.379e-17 | 1.289e-16 | 2.5454e-14 | True |
| core/cb384647987dbe4e__hsd | unchanged_working_update | 28 | 1.724e-09 | 1.724e-09 | 6.137e-09 | 3.0261e-08 | True |
| core/cb384647987dbe4e__hss | iteration_limit | 30 | 1.868e-08 | 1.868e-08 | 7.585e-08 | 1.6535e-05 | False |
| core/cb384647987dbe4e__input16_ssd | working_precision_cycle | 7 | 1.759e-09 | 1.759e-09 | 6.985e-09 | 3.0261e-08 | True |
| core/cb384647987dbe4e__sdd | iteration_limit | 30 | 2.225e-17 | 3.134e-17 | 1.089e-16 | 1.8251e-14 | True |
| core/cb384647987dbe4e__ssd | unchanged_working_update | 3 | 1.500e-09 | 1.500e-09 | 5.774e-09 | 3.0261e-08 | True |
| core/cb384647987dbe4e__sss | iteration_limit | 30 | 1.436e-08 | 1.436e-08 | 5.289e-08 | 9.9003e-06 | False |
| cuda_bridge/cb384647987dbe4e__blocked32_w32_r64 | unchanged_working_update | 2 | 1.500e-09 | 1.500e-09 | 5.774e-09 | 3.0261e-08 | True |
| cuda_bridge/cb384647987dbe4e__mixed16inputs_w32_r64 | unchanged_working_update | 6 | 1.512e-09 | 1.512e-09 | 5.824e-09 | 3.0261e-08 | True |
| cuda_bridge/cb384647987dbe4e__native32_w32_r32 | iteration_limit | 30 | 9.058e-09 | 9.058e-09 | 3.939e-08 | 8.6000e-06 | False |
| cuda_bridge/cb384647987dbe4e__native32_w32_r64 | unchanged_working_update | 3 | 1.500e-09 | 1.500e-09 | 5.774e-09 | 3.0261e-08 | True |

Recorded prior metrics (final dicts) are stored alongside in teaching_case_replay.json for comparison; the prior evidence (CONTEXT.md) states hsd final forward error 3.0260721054e-8 and hss 1.6534729676e-5.