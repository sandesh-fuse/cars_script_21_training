# Feature ablation results

Baseline vs. each group, tested independently (not cumulative).


## OVERALL

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 13,885 | $515 | $889 | 0.6009 | 90.2% | $2,317 | $-111 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 13,885 | $549 | $941 | 0.6313 | 88.6% | $2,261 | $-42 |
| group_b_gvm_engine (+gvm_range/engine_type) | 13,885 | $549 | $935 | 0.6338 | 88.8% | $2,298 | $-28 |
| group_c_tonnage (+tonnage) | 13,885 | $549 | $940 | 0.6323 | 88.9% | $2,288 | $-30 |

## $0-200

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 573 | $222 | $333 | 1.1358 | 63.0% | $1,103 | $+218 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 573 | $263 | $361 | 1.2303 | 58.5% | $1,198 | $+261 |
| group_b_gvm_engine (+gvm_range/engine_type) | 573 | $267 | $361 | 1.2376 | 57.9% | $1,207 | $+264 |
| group_c_tonnage (+tonnage) | 573 | $265 | $360 | 1.2348 | 56.5% | $1,199 | $+263 |

## $200-500

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 4,868 | $230 | $374 | 0.5611 | 88.8% | $1,386 | $+203 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 4,868 | $263 | $417 | 0.6067 | 86.8% | $1,423 | $+246 |
| group_b_gvm_engine (+gvm_range/engine_type) | 4,868 | $272 | $428 | 0.6182 | 86.5% | $1,451 | $+256 |
| group_c_tonnage (+tonnage) | 4,868 | $268 | $424 | 0.6125 | 87.1% | $1,441 | $+251 |

## $500-1K

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 3,430 | $332 | $522 | 0.4861 | 97.0% | $2,000 | $+154 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 3,430 | $359 | $572 | 0.4994 | 96.5% | $1,990 | $+204 |
| group_b_gvm_engine (+gvm_range/engine_type) | 3,430 | $366 | $586 | 0.5050 | 96.3% | $2,027 | $+218 |
| group_c_tonnage (+tonnage) | 3,430 | $364 | $583 | 0.5037 | 96.6% | $2,017 | $+213 |

## $1K-2.5K

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 3,354 | $585 | $762 | 0.5387 | 94.8% | $2,850 | $-228 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 3,354 | $620 | $845 | 0.5443 | 94.3% | $2,742 | $-164 |
| group_b_gvm_engine (+gvm_range/engine_type) | 3,354 | $617 | $840 | 0.5372 | 94.9% | $2,779 | $-148 |
| group_c_tonnage (+tonnage) | 3,354 | $620 | $846 | 0.5408 | 94.7% | $2,766 | $-148 |

## $2.5K-4K

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 984 | $1,232 | $1,452 | 0.7098 | 84.9% | $4,265 | $-988 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 984 | $1,293 | $1,554 | 0.7303 | 83.2% | $3,990 | $-848 |
| group_b_gvm_engine (+gvm_range/engine_type) | 984 | $1,277 | $1,537 | 0.7219 | 84.2% | $4,036 | $-830 |
| group_c_tonnage (+tonnage) | 984 | $1,289 | $1,556 | 0.7258 | 83.8% | $4,034 | $-817 |

## $4K-6K

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 453 | $2,002 | $2,313 | 0.7843 | 76.6% | $5,643 | $-1,783 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 453 | $2,073 | $2,410 | 0.8114 | 70.9% | $5,194 | $-1,552 |
| group_b_gvm_engine (+gvm_range/engine_type) | 453 | $2,029 | $2,371 | 0.8020 | 71.5% | $5,286 | $-1,520 |
| group_c_tonnage (+tonnage) | 453 | $2,039 | $2,376 | 0.7985 | 72.0% | $5,266 | $-1,506 |

## $6K-10K

| group | N | MAE | RMSE | RMSLE | Cov90 | Width | Bias |
|---|---|---|---|---|---|---|---|
| Baseline (none) | 223 | $3,048 | $3,574 | 0.9100 | 66.4% | $7,284 | $-2,853 |
| group_a_mileage_title (+true_mileage_unknown/clean_title) | 223 | $2,979 | $3,580 | 0.9507 | 60.5% | $6,658 | $-2,432 |
| group_b_gvm_engine (+gvm_range/engine_type) | 223 | $2,900 | $3,523 | 0.9304 | 65.0% | $6,817 | $-2,388 |
| group_c_tonnage (+tonnage) | 223 | $2,939 | $3,544 | 0.9313 | 62.8% | $6,798 | $-2,388 |
