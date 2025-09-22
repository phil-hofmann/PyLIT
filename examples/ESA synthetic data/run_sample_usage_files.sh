#!/bin/bash
kernel_name="Gauss" # Gauss or Uniform
method_name="cdf_l2_fit" # cdf_l2_fit, max_entropy_fit, l2_fit
kindex="4"
poetry run python sample_usage_chi2kink.py ${kernel_name} \
                                ${method_name} \
                                synthetic_example/inputs/F_synthetic_noise_0p1_k${kindex}_rs10.csv \
                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/Skw_${kernel_name}_${method_name}_synthetic_noise_0p1_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/SSE_${kernel_name}_${method_name}_synthetic_noise_0p1_k${kindex}_rs10.csv

poetry run python sample_usage_chi2kink.py ${kernel_name} \
                                ${method_name} \
                                synthetic_example/inputs/F_synthetic_noise_0p01_k${kindex}_rs10.csv \
                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/Skw_${kernel_name}_${method_name}_synthetic_noise_0p01_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/SSE_${kernel_name}_${method_name}_synthetic_noise_0p01_k${kindex}_rs10.csv

poetry run python sample_usage_chi2kink.py ${kernel_name} \
                                ${method_name} \
                                synthetic_example/inputs/F_synthetic_noise_0p001_k${kindex}_rs10.csv \
                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/Skw_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv \
                                synthetic_example/outputs/chi2kink/SSE_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv

#poetry run python sample_usage_Bayesian.py ${kernel_name} \
#                                ${method_name} \
#                                synthetic_example/inputs/F_synthetic_noise_0p1_k${kindex}_rs10.csv \
#                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/Skw_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/posterior_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv

#poetry run python sample_usage_Bayesian.py ${kernel_name} \
#                                ${method_name} \
#                                synthetic_example/inputs/F_synthetic_noise_0p01_k${kindex}_rs10.csv \
#                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/Skw_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/posterior_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv

#poetry run python sample_usage_Bayesian.py ${kernel_name} \
#                                ${method_name} \
#                                synthetic_example/inputs/F_synthetic_noise_0p001_k${kindex}_rs10.csv \
#                                synthetic_example/inputs/D_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/Skw_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv \
#                                synthetic_example/outputs/Bayesian/posterior_${kernel_name}_${method_name}_synthetic_noise_0p001_k${kindex}_rs10.csv

poetry run python plot_noise_convergence.py ${kernel_name} ${method_name} ${kindex}
