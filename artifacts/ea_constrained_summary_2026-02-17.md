# Constrained EA Summary (2026-02-17)

Constraint set:
- hard caps: rapa<=0.80, nad<=0.90, seno<=0.80, yama<=0.15, transplant<=0.80, exercise<=0.60
- penalty: 0.020*total_dose + 0.150*yamanaka + 0.500*cap_violation

## Profile comparison (constrained raw fitness vs unconstrained)
- default: raw_delta=-0.033479, ATP_delta=-0.0209, het_delta=+0.0251
- near_cliff_80: raw_delta=-0.034055, ATP_delta=-0.0203, het_delta=+0.0275
- post_chemo_55: raw_delta=-0.034672, ATP_delta=-0.0216, het_delta=+0.0262
- melas_35: raw_delta=-0.031176, ATP_delta=-0.0173, het_delta=+0.0278
