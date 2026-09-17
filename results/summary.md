# SinkProbe results summary

## hf
- **Qwen3-0.6B-Base**: gated_layers 0
- **Qwen3.5-0.8B-Base**: gated_layers 6

## hf_compare
- **Qwen3.5-0.8B-Base minus Qwen3-0.6B-Base @2048**: recency_gap_diff 0.120 ± 0.000; recall_diff 0.07954545454545459
- **Qwen3.5-0.8B-Base minus Qwen3-0.6B-Base @4096**: recency_gap_diff -0.179 ± -0.411; recall_diff 0.13636363636363635
- **Qwen3.5-0.8B-Base minus Qwen3-0.6B-Base @8192**: recency_gap_diff -0.210 ± -0.425; recall_diff 0.03409090909090917
- **Qwen3.5-0.8B-Base minus Qwen3-0.6B-Base @16384**: recency_gap_diff -0.122 ± -0.303; recall_diff 0.03409090909090917

## learnability
- **softmax**: learned 2; trained 2; recall_by_seed {'0': 1.0, '1': 1.0}; long_recall_learned 9.155 ± 23.266
- **gate**: learned 2; trained 2; recall_by_seed {'0': 0.990966796875, '1': 1.0}; long_recall_learned 4.382 ± 46.066
- **hybrid**: learned 1; trained 2; recall_by_seed {'0': 0.98974609375, '1': 0.69091796875}; long_recall_learned 17.139 ± nan
- **hybrid_nope**: learned 1; trained 2; recall_by_seed {'0': 0.008544921875, '1': 1.0}; long_recall_learned 87.402 ± nan
- **hybrid_attnres**: learned 1; trained 2; recall_by_seed {'0': 0.998779296875, '1': 0.0078125}; long_recall_learned 95.337 ± nan
- **hybrid_nogate**: learned 1; trained 2; recall_by_seed {'0': 1.0, '1': 0.00537109375}; long_recall_learned 19.946 ± nan
- **nope**: learned 1; trained 1; recall_by_seed {'0': 0.99951171875}; long_recall_learned 0.000 ± nan
- **attnres**: learned 1; trained 1; recall_by_seed {'0': 1.0}; long_recall_learned 11.572 ± nan
- **sinklogit**: learned 1; trained 1; recall_by_seed {'0': 1.0}; long_recall_learned 2.930 ± nan
- **softpick**: learned 1; trained 1; recall_by_seed {'0': 0.9990234375}; long_recall_learned 0.146 ± nan
- **hybrid_nodelta**: learned 0; trained 1; recall_by_seed {'0': 0.15673828125}; long_recall_learned nan ± nan

## ladder
- **softmax**: recall@256 100.000 ± 0.000; recall@2048 9.155 ± 23.266; sink_mass@256 0.041 ± 0.073; sink_ratio@256 2.041 ± 3.615; sink_rate@256 0.016 ± 0.199; sink_noop@256 0.054 ± 0.104; sink_copy@256 0.023 ± 0.044; sink_answer@256 0.010 ± 0.012; act_max@256 477.735 ± 2148.738; recency_gap@256 0.000 ± 0.000; recency_gap@2048 31.446 ± 115.583; middle_dip@2048 14.750 ± 63.391; n 2; noop_over_copy@256 2.316 ± 0.062
- **gate**: recall@256 99.548 ± 5.739; recall@2048 4.382 ± 46.066; sink_mass@256 0.053 ± 0.019; sink_ratio@256 2.621 ± 0.954; sink_rate@256 0.016 ± 0.199; sink_noop@256 0.059 ± 0.075; sink_copy@256 0.048 ± 0.072; sink_answer@256 0.007 ± 0.015; act_max@256 251.289 ± 123.536; gate_mean@256 0.335 ± 0.096; gate_noop@256 0.324 ± 0.170; gate_copy@256 0.353 ± 0.009; recency_gap@256 0.021 ± 0.265; recency_gap@2048 14.337 ± 171.154; middle_dip@2048 6.284 ± 73.277; n 2; noop_over_copy@256 1.268 ± 3.459
- **hybrid**: recall@256 98.975 ± nan; recall@2048 17.139 ± nan; sink_mass@256 0.009 ± nan; sink_ratio@256 0.431 ± nan; sink_rate@256 0.000 ± nan; sink_noop@256 0.012 ± nan; sink_copy@256 0.002 ± nan; sink_answer@256 0.002 ± nan; act_max@256 504.951 ± nan; gate_mean@256 0.488 ± nan; gate_noop@256 0.476 ± nan; gate_copy@256 0.498 ± nan; recency_gap@256 0.355 ± nan; recency_gap@2048 62.524 ± nan; middle_dip@2048 29.416 ± nan; n 1; noop_over_copy@256 5.286 ± nan
- **hybrid_nope**: recall@256 100.000 ± nan; recall@2048 87.402 ± nan; sink_mass@256 0.029 ± nan; sink_ratio@256 1.422 ± nan; sink_rate@256 0.000 ± nan; sink_noop@256 0.041 ± nan; sink_copy@256 0.012 ± nan; sink_answer@256 0.006 ± nan; act_max@256 721.625 ± nan; gate_mean@256 0.541 ± nan; gate_noop@256 0.534 ± nan; gate_copy@256 0.542 ± nan; recency_gap@256 0.000 ± nan; recency_gap@2048 -1.102 ± nan; middle_dip@2048 0.675 ± nan; n 1; noop_over_copy@256 3.305 ± nan
- **hybrid_attnres**: recall@256 99.878 ± nan; recall@2048 95.337 ± nan; sink_mass@256 0.022 ± nan; sink_ratio@256 1.091 ± nan; sink_rate@256 0.000 ± nan; sink_noop@256 0.031 ± nan; sink_copy@256 0.006 ± nan; sink_answer@256 0.003 ± nan; act_max@256 165.241 ± nan; gate_mean@256 0.520 ± nan; gate_noop@256 0.532 ± nan; gate_copy@256 0.499 ± nan; recency_gap@256 -0.068 ± nan; recency_gap@2048 0.486 ± nan; middle_dip@2048 0.758 ± nan; n 1; noop_over_copy@256 5.545 ± nan
- **sinklogit**: recall@256 100.000 ± nan; recall@2048 2.930 ± nan; sink_mass@256 0.057 ± nan; sink_ratio@256 2.828 ± nan; sink_rate@256 0.031 ± nan; sink_noop@256 0.070 ± nan; sink_copy@256 0.036 ± nan; sink_answer@256 0.020 ± nan; act_max@256 593.455 ± nan; virtual_sink@256 0.017 ± nan; recency_gap@256 0.000 ± nan; recency_gap@2048 7.328 ± nan; middle_dip@2048 3.441 ± nan; n 1; noop_over_copy@256 1.959 ± nan
- **softpick**: recall@256 99.902 ± nan; recall@2048 0.146 ± nan; sink_mass@256 0.047 ± nan; sink_ratio@256 2.354 ± nan; sink_rate@256 0.062 ± nan; sink_noop@256 0.069 ± nan; sink_copy@256 0.015 ± nan; sink_answer@256 0.010 ± nan; act_max@256 329.797 ± nan; recency_gap@256 0.382 ± nan; recency_gap@2048 0.289 ± nan; middle_dip@2048 -0.004 ± nan; n 1; noop_over_copy@256 4.488 ± nan

## steps
- **gate:sink_ratio@256**: diff 0.5801135199950469; ci 2.8176119802177175; p 0.27451192724047097
- **gate:sink_noop@256**: diff 0.005234300836700442; ci 0.04764413312135072; p 0.6588041808386731
- **gate:recall@256**: diff -0.45166015625; ci 5.738886416485152; p 0.5000000000000001
- **gate:recall@2048**: diff -4.77294921875; ci 24.92974794112807; p 0.39510921172553687
- **gate:recency_gap@2048**: diff -17.109725275609698; ci 80.17372451209397; p 0.41576140365681774

## dose

## skew

## dissociation
- **sink_ratio~recency_gap@256**: 0.078 ± -0.591
- **sink_ratio~middle_dip@256**: 0.335 ± -0.359
- **sink_ratio~recall@256**: 0.351 ± -0.370
- **sink_ratio~recency_gap@2048**: 0.119 ± -0.614
- **sink_ratio~middle_dip@2048**: 0.000 ± -0.670
- **sink_ratio~recall@2048**: -0.448 ± -0.870

## noop_enrichment
- **runs**: 12
- **noop_higher**: 11
- **sign_test_p**: 0.00634765625
- **wilcoxon_p**: 0.0009765625
- **mean_diff**: 0.040 ± 0.021
- **median_ratio**: 2.7183397238044593

## mechanisms
- **gate**: params 1870720; n 2; sink_ratio@256 {'diff': 0.5801135199950469, 'ci': 2.8176119802177175, 'p': 0.27451192724047097}; sink_noop@256 {'diff': 0.005234300836700442, 'ci': 0.04764413312135072, 'p': 0.6588041808386731}; recall@256 {'diff': -0.45166015625, 'ci': 5.738886416485152, 'p': 0.5000000000000001}; recall@2048 {'diff': -4.77294921875, 'ci': 24.92974794112807, 'p': 0.39510921172553687}; fit_slope@2048 {'diff': -21.066893892973255, 'ci': 103.43531391320538, 'p': 0.42801938605499007}; fit_curv@2048 {'diff': -62.040701505595536, 'ci': 301.12466129675045, 'p': 0.4622969271865018}; act_max@256 {'diff': -226.44589102268213, 'ci': 2119.170537504708, 'p': 0.40778827241776655}

## warmup

## perplexity_check
- **Qwen3-0.6B-Base**: [{'bias': 0.0, 'nll': 2.557866232598363, 'ppl': 12.908244717458542}, {'bias': -inf, 'nll': 2.5579010659976253, 'ppl': 12.908694363331863, 'nll_diff': [3.4833399262329534e-05, 0.001293590785688586]}]
- **Qwen3.5-0.8B-Base**: [{'bias': 0.0, 'nll': 2.5905766729988717, 'ppl': 13.337460739405996}, {'bias': -inf, 'nll': 2.6060938428762404, 'ppl': 13.546034435107245, 'nll_diff': [0.015517169877368231, 0.002665726453995669]}]

## interventions_synthetic
- **softmax@256:bias0**: n 2; sink_mass 0.041 ± 0.071; sink_mass_diff 0.000 ± 0.000; sink_noop 0.053 ± 0.092; sink_noop_diff 0.000 ± 0.000; recall 100.000 ± 0.000; recall_diff 0.000 ± 0.000; recency_gap 0.000 ± 0.000; recency_gap_diff 0.000 ± 0.000; fit_slope 0.000 ± 0.000; fit_slope_diff 0.000 ± 0.000; fit_curv -0.000 ± 0.000; fit_curv_diff 0.000 ± 0.000
- **softmax@1024:bias0**: n 2; sink_mass 0.013 ± 0.028; sink_mass_diff 0.000 ± 0.000; sink_noop 0.018 ± 0.041; sink_noop_diff 0.000 ± 0.000; recall 45.166 ± 5.584; recall_diff 0.000 ± 0.000; recency_gap 90.460 ± 31.201; recency_gap_diff 0.000 ± 0.000; fit_slope 121.581 ± 50.900; fit_slope_diff 0.000 ± 0.000; fit_curv 95.693 ± 129.188; fit_curv_diff 0.000 ± 0.000
- **softmax@256:bias-inf**: n 2; sink_mass 0.000 ± 0.000; sink_mass_diff -0.041 ± 0.071; sink_noop 0.000 ± 0.000; sink_noop_diff -0.053 ± 0.092; recall 97.900 ± 15.511; recall_diff -2.100 ± 15.511; recency_gap 7.862 ± 56.909; recency_gap_diff 7.862 ± 56.909; fit_slope 11.044 ± 80.220; fit_slope_diff 11.044 ± 80.220; fit_curv -40.845 ± 298.877; fit_curv_diff -40.845 ± 298.877
- **softmax@1024:bias-inf**: n 2; sink_mass 0.000 ± 0.000; sink_mass_diff -0.013 ± 0.028; sink_noop 0.000 ± 0.000; sink_noop_diff -0.018 ± 0.041; recall 45.190 ± 7.135; recall_diff 0.024 ± 1.551; recency_gap 91.613 ± 26.448; recency_gap_diff 1.153 ± 4.753; fit_slope 122.979 ± 43.132; fit_slope_diff 1.399 ± 7.769; fit_curv 86.985 ± 120.505; fit_curv_diff -8.708 ± 8.683
- **gate@256:bias0**: n 2; sink_mass 0.051 ± 0.016; sink_mass_diff 0.000 ± 0.000; sink_noop 0.058 ± 0.052; sink_noop_diff 0.000 ± 0.000; recall 99.512 ± 6.204; recall_diff 0.000 ± 0.000; recency_gap 0.298 ± 3.785; recency_gap_diff 0.000 ± 0.000; fit_slope 0.043 ± 0.543; fit_slope_diff 0.000 ± 0.000; fit_curv 0.411 ± 5.216; fit_curv_diff 0.000 ± 0.000
- **gate@1024:bias0**: n 2; sink_mass 0.017 ± 0.024; sink_mass_diff 0.000 ± 0.000; sink_noop 0.018 ± 0.013; sink_noop_diff 0.000 ± 0.000; recall 27.759 ± 211.873; recall_diff 0.000 ± 0.000; recency_gap 60.651 ± 415.488; recency_gap_diff 0.000 ± 0.000; fit_slope 78.371 ± 561.619; fit_slope_diff 0.000 ± 0.000; fit_curv 78.325 ± 285.247; fit_curv_diff 0.000 ± 0.000
- **gate@256:bias-inf**: n 2; sink_mass 0.000 ± 0.000; sink_mass_diff -0.051 ± 0.016; sink_noop 0.000 ± 0.000; sink_noop_diff -0.058 ± 0.052; recall 94.727 ± 57.699; recall_diff -4.785 ± 51.495; recency_gap 17.502 ± 186.561; recency_gap_diff 17.204 ± 182.777; fit_slope 23.712 ± 251.199; fit_slope_diff 23.670 ± 250.656; fit_curv -80.639 ± 841.187; fit_curv_diff -81.049 ± 846.403
- **gate@1024:bias-inf**: n 2; sink_mass 0.000 ± 0.000; sink_mass_diff -0.017 ± 0.024; sink_noop 0.000 ± 0.000; sink_noop_diff -0.018 ± 0.013; recall 28.052 ± 214.976; recall_diff 0.293 ± 3.102; recency_gap 60.735 ± 411.721; recency_gap_diff 0.084 ± 3.767; fit_slope 79.006 ± 563.429; fit_slope_diff 0.635 ± 1.810; fit_curv 77.713 ± 287.469; fit_curv_diff -0.612 ± 2.222
- **hybrid@256:bias0**: n 1; sink_mass 0.009 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.012 ± nan; sink_noop_diff 0.000 ± nan; recall 99.121 ± nan; recall_diff 0.000 ± nan; recency_gap 0.596 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.312 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -1.017 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid@1024:bias0**: n 1; sink_mass 0.003 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.004 ± nan; sink_noop_diff 0.000 ± nan; recall 36.328 ± nan; recall_diff 0.000 ± nan; recency_gap 87.135 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 118.288 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 110.665 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.009 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.012 ± nan; recall 99.121 ± nan; recall_diff 0.000 ± nan; recency_gap 0.596 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.312 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -1.017 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.003 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.004 ± nan; recall 36.182 ± nan; recall_diff -0.146 ± nan; recency_gap 86.757 ± nan; recency_gap_diff -0.378 ± nan; fit_slope 117.924 ± nan; fit_slope_diff -0.364 ± nan; fit_curv 109.875 ± nan; fit_curv_diff -0.791 ± nan
- **hybrid_nope@256:bias0**: n 1; sink_mass 0.029 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.039 ± nan; sink_noop_diff 0.000 ± nan; recall 100.000 ± nan; recall_diff 0.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -0.000 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nope@1024:bias0**: n 1; sink_mass 0.009 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.013 ± nan; sink_noop_diff 0.000 ± nan; recall 88.379 ± nan; recall_diff 0.000 ± nan; recency_gap 1.510 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 1.384 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 10.415 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nope@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.029 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.039 ± nan; recall 100.000 ± nan; recall_diff 0.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -0.000 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nope@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.009 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.013 ± nan; recall 88.477 ± nan; recall_diff 0.098 ± nan; recency_gap 1.692 ± nan; recency_gap_diff 0.182 ± nan; fit_slope 1.543 ± nan; fit_slope_diff 0.159 ± nan; fit_curv 10.832 ± nan; fit_curv_diff 0.416 ± nan
- **hybrid_attnres@256:bias0**: n 1; sink_mass 0.021 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.030 ± nan; sink_noop_diff 0.000 ± nan; recall 99.854 ± nan; recall_diff 0.000 ± nan; recency_gap 0.010 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -0.059 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -0.390 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_attnres@1024:bias0**: n 1; sink_mass 0.008 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.011 ± nan; sink_noop_diff 0.000 ± nan; recall 98.584 ± nan; recall_diff 0.000 ± nan; recency_gap -0.174 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -0.391 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 0.139 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_attnres@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.021 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.030 ± nan; recall 99.854 ± nan; recall_diff 0.000 ± nan; recency_gap 0.010 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -0.059 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -0.390 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_attnres@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.008 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.011 ± nan; recall 98.584 ± nan; recall_diff 0.000 ± nan; recency_gap -0.174 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -0.391 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 0.139 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nogate@256:bias0**: n 1; sink_mass 0.084 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.131 ± nan; sink_noop_diff 0.000 ± nan; recall 99.902 ± nan; recall_diff 0.000 ± nan; recency_gap 0.010 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -0.087 ± nan; fit_slope_diff 0.000 ± nan; fit_curv -0.629 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nogate@1024:bias0**: n 1; sink_mass 0.047 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.071 ± nan; sink_noop_diff 0.000 ± nan; recall 42.822 ± nan; recall_diff 0.000 ± nan; recency_gap 98.662 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 136.028 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 107.014 ± nan; fit_curv_diff 0.000 ± nan
- **hybrid_nogate@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.084 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.131 ± nan; recall 98.047 ± nan; recall_diff -1.855 ± nan; recency_gap 7.260 ± nan; recency_gap_diff 7.249 ± nan; fit_slope 10.089 ± nan; fit_slope_diff 10.176 ± nan; fit_curv -36.700 ± nan; fit_curv_diff -36.071 ± nan
- **hybrid_nogate@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.047 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.071 ± nan; recall 43.018 ± nan; recall_diff 0.195 ± nan; recency_gap 98.470 ± nan; recency_gap_diff -0.191 ± nan; fit_slope 136.144 ± nan; fit_slope_diff 0.117 ± nan; fit_curv 105.842 ± nan; fit_curv_diff -1.172 ± nan
- **nope@256:bias0**: n 1; sink_mass 0.071 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.091 ± nan; sink_noop_diff 0.000 ± nan; recall 100.000 ± nan; recall_diff 0.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 0.000 ± nan; fit_curv_diff 0.000 ± nan
- **nope@1024:bias0**: n 1; sink_mass 0.054 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.061 ± nan; sink_noop_diff 0.000 ± nan; recall 7.031 ± nan; recall_diff 0.000 ± nan; recency_gap -26.195 ± nan; recency_gap_diff 0.000 ± nan; fit_slope -31.040 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 73.644 ± nan; fit_curv_diff 0.000 ± nan
- **nope@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.071 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.091 ± nan; recall 0.000 ± nan; recall_diff -100.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff -0.000 ± nan; fit_curv 0.000 ± nan; fit_curv_diff -0.000 ± nan
- **nope@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.054 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.061 ± nan; recall 0.000 ± nan; recall_diff -7.031 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 26.195 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 31.040 ± nan; fit_curv 0.000 ± nan; fit_curv_diff -73.644 ± nan
- **attnres@256:bias0**: n 1; sink_mass 0.064 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.088 ± nan; sink_noop_diff 0.000 ± nan; recall 100.000 ± nan; recall_diff 0.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 0.000 ± nan; fit_curv_diff 0.000 ± nan
- **attnres@1024:bias0**: n 1; sink_mass 0.020 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.028 ± nan; sink_noop_diff 0.000 ± nan; recall 52.637 ± nan; recall_diff 0.000 ± nan; recency_gap 90.255 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 125.274 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 15.784 ± nan; fit_curv_diff 0.000 ± nan
- **attnres@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.064 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.088 ± nan; recall 96.338 ± nan; recall_diff -3.662 ± nan; recency_gap 13.612 ± nan; recency_gap_diff 13.612 ± nan; fit_slope 19.145 ± nan; fit_slope_diff 19.145 ± nan; fit_curv -70.993 ± nan; fit_curv_diff -70.993 ± nan
- **attnres@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.020 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.028 ± nan; recall 52.051 ± nan; recall_diff -0.586 ± nan; recency_gap 91.020 ± nan; recency_gap_diff 0.765 ± nan; fit_slope 126.324 ± nan; fit_slope_diff 1.051 ± nan; fit_curv 15.939 ± nan; fit_curv_diff 0.156 ± nan
- **sinklogit@256:bias0**: n 1; sink_mass 0.056 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.070 ± nan; sink_noop_diff 0.000 ± nan; recall 100.000 ± nan; recall_diff 0.000 ± nan; recency_gap 0.000 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 0.000 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 0.000 ± nan; fit_curv_diff 0.000 ± nan
- **sinklogit@1024:bias0**: n 1; sink_mass 0.018 ± nan; sink_mass_diff 0.000 ± nan; sink_noop 0.023 ± nan; sink_noop_diff 0.000 ± nan; recall 41.748 ± nan; recall_diff 0.000 ± nan; recency_gap 75.864 ± nan; recency_gap_diff 0.000 ± nan; fit_slope 98.858 ± nan; fit_slope_diff 0.000 ± nan; fit_curv 91.096 ± nan; fit_curv_diff 0.000 ± nan
- **sinklogit@256:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.056 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.070 ± nan; recall 95.898 ± nan; recall_diff -4.102 ± nan; recency_gap 13.098 ± nan; recency_gap_diff 13.098 ± nan; fit_slope 18.445 ± nan; fit_slope_diff 18.445 ± nan; fit_curv -64.902 ± nan; fit_curv_diff -64.902 ± nan
- **sinklogit@1024:bias-inf**: n 1; sink_mass 0.000 ± nan; sink_mass_diff -0.018 ± nan; sink_noop 0.000 ± nan; sink_noop_diff -0.023 ± nan; recall 39.502 ± nan; recall_diff -2.246 ± nan; recency_gap 77.033 ± nan; recency_gap_diff 1.169 ± nan; fit_slope 101.670 ± nan; fit_slope_diff 2.812 ± nan; fit_curv 84.310 ± nan; fit_curv_diff -6.785 ± nan
- **pooled**: with_positional_signal@256 {'recall_diff': (-2.3388671875, 1.986685941230054, 10), 'n_models': 10}; nope_softmax@256 {'recall_diff': (-100.0, nan, 1), 'n_models': 1}; with_positional_signal@1024 {'recall_diff': (-0.205078125, 0.5519898332708311, 10), 'n_models': 10}; nope_softmax@1024 {'recall_diff': (-7.03125, nan, 1), 'n_models': 1}

## interventions_hf
- **Qwen3-0.6B-Base@2048:bias0**: n 44; recall 0.8181818181818182; recall_ci 0.680 ± 0.905; recall_diff 0.000 ± 0.000; logprob -0.579 ± 0.050; logprob_diff 0.000 ± 0.000; sink_mass 0.471 ± 0.009
- **Qwen3-0.6B-Base@2048:bias-inf**: n 44; recall 0.9090909090909091; recall_ci 0.788 ± 0.964; recall_diff 0.091 ± 0.023; logprob -0.551 ± 0.052; logprob_diff 0.028 ± 0.017; sink_mass 0.000 ± 0.000
- **Qwen3-0.6B-Base@4096:bias0**: n 44; recall 0.8409090909090909; recall_ci 0.706 ± 0.921; recall_diff 0.000 ± 0.000; logprob -0.611 ± 0.064; logprob_diff 0.000 ± 0.000; sink_mass 0.458 ± 0.008
- **Qwen3-0.6B-Base@4096:bias-inf**: n 44; recall 0.9772727272727273; recall_ci 0.882 ± 0.996; recall_diff 0.136 ± 0.045; logprob -0.530 ± 0.053; logprob_diff 0.081 ± 0.028; sink_mass 0.000 ± 0.000
- **Qwen3.5-0.8B-Base@2048:bias0**: n 44; recall 1.0; recall_ci 0.920 ± 1.000; recall_diff 0.000 ± 0.000; logprob -0.100 ± 0.017; logprob_diff 0.000 ± 0.000; sink_mass 0.039 ± 0.003
- **Qwen3.5-0.8B-Base@2048:bias-inf**: n 44; recall 1.0; recall_ci 0.920 ± 1.000; recall_diff 0.000 ± 0.000; logprob -0.108 ± 0.028; logprob_diff -0.008 ± 0.014; sink_mass 0.000 ± 0.000
- **Qwen3.5-0.8B-Base@4096:bias0**: n 44; recall 0.9545454545454546; recall_ci 0.849 ± 0.987; recall_diff 0.000 ± 0.000; logprob -0.342 ± 0.274; logprob_diff 0.000 ± 0.000; sink_mass 0.033 ± 0.002
- **Qwen3.5-0.8B-Base@4096:bias-inf**: n 44; recall 0.9318181818181818; recall_ci 0.818 ± 0.977; recall_diff -0.023 ± -0.068; logprob -0.409 ± 0.295; logprob_diff -0.066 ± 0.044; sink_mass 0.000 ± 0.000
