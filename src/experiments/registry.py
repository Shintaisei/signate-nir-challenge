"""登録済み実験一覧。新しい実験はここに1行追加する。"""

from __future__ import annotations

from dataclasses import dataclass
from pipeline.models import get_predictor
from pipeline.preprocessors import get_preprocessor
from pipeline.types import Predictor, Preprocessor


@dataclass(frozen=True)
class ExperimentSpec:
    preprocessor: str
    predictor: str
    memo: str


def _build(spec: ExperimentSpec) -> tuple[Preprocessor, Predictor, str]:
    return get_preprocessor(spec.preprocessor), get_predictor(spec.predictor), spec.memo


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline_mean": ExperimentSpec(
        preprocessor="spectral",
        predictor="mean",
        memo="baseline: train mean moisture (spectral features ignored by mean)",
    ),
    "candidate_linear_1f": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_linear",
        memo="candidate1: best single wavelength linear regression",
    ),
    "candidate_linear_1f_rank2": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_second_wavelength",
        memo="raw: 2nd-best correlated wavelength linear regression",
    ),
    "candidate_linear_1f_stable_wavelength": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_stable_wavelength",
        memo="raw: stable top-1 wavelength across train chunks",
    ),
    "candidate_linear_1f_loo_rmse": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_loo_rmse_wavelength",
        memo="raw: wavelength minimizing LOO species RMSE",
    ),
    "candidate_linear_1f_top3_median": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_top3_median",
        memo="raw: median of top-3 correlated wavelength linear models",
    ),
    "candidate_linear_1f_oof_intercept": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_linear_oof_intercept",
        memo="raw: best wavelength + LOO species OOF intercept correction",
    ),
    "candidate_linear_1f_msc": ExperimentSpec(
        preprocessor="spectral_msc",
        predictor="single_feature_linear",
        memo="raw MSC + best single wavelength linear regression",
    ),
    "candidate_linear_1f_area": ExperimentSpec(
        preprocessor="spectral_area",
        predictor="single_feature_linear",
        memo="raw area normalization + best single wavelength linear regression",
    ),
    "candidate_linear_1f_smooth": ExperimentSpec(
        preprocessor="spectral_smooth",
        predictor="single_feature_linear",
        memo="raw smooth + best single wavelength linear regression",
    ),
    "candidate_linear_1f_detrend": ExperimentSpec(
        preprocessor="spectral_detrend",
        predictor="single_feature_linear",
        memo="raw detrend + best single wavelength linear regression",
    ),
    "candidate_linear_1f_range_cal": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_linear_range_calibrated",
        memo="raw 1f + LOO species OOF affine range calibration",
    ),
    "candidate_linear_1f_lowtail_mild": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_linear_low_tail_mild",
        memo="raw 1f + mild low-tail extension",
    ),
    "candidate_linear_1f_moisture_bins": ExperimentSpec(
        preprocessor="spectral",
        predictor="moisture_bin_wavelength_linear",
        memo="raw: moisture quartile bins, 1 wavelength per bin",
    ),
    "candidate_linear_1f_nn_bias": ExperimentSpec(
        preprocessor="spectral",
        predictor="nearest_train_species_bias_corrected",
        memo="raw 1f + nearest-train-species mean residual correction",
    ),
    "candidate_linear_1f_moisture_bins_v2": ExperimentSpec(
        preprocessor="spectral",
        predictor="moisture_bin_fixed_wavelength",
        memo="raw: fixed global wavelength + moisture-bin slopes (shrunk)",
    ),
    "candidate_linear_1f_msc_select_raw": ExperimentSpec(
        preprocessor="spectral",
        predictor="msc_select_raw_linear",
        memo="MSC wavelength selection + raw spectrum linear regression",
    ),
    "candidate_linear_1f_center": ExperimentSpec(
        preprocessor="spectral_center",
        predictor="single_feature_linear",
        memo="row center only + best single wavelength",
    ),
    "candidate_linear_1f_l2norm": ExperimentSpec(
        preprocessor="spectral_l2norm",
        predictor="single_feature_linear",
        memo="L2 norm per spectrum + best single wavelength",
    ),
    "candidate_linear_1f_blend_snv25": ExperimentSpec(
        preprocessor="spectral_blend_snv25",
        predictor="single_feature_linear",
        memo="75% raw + 25% SNV blend + best single wavelength",
    ),
    "candidate_linear_1f_blend_snv50": ExperimentSpec(
        preprocessor="spectral_blend_snv50",
        predictor="single_feature_linear",
        memo="50% raw + 50% SNV blend + best single wavelength",
    ),
    "candidate_linear_1f_smooth3": ExperimentSpec(
        preprocessor="spectral_smooth3",
        predictor="single_feature_linear",
        memo="light smooth (w=3) + best single wavelength",
    ),
    "candidate_linear_1f_smooth5": ExperimentSpec(
        preprocessor="spectral_smooth5",
        predictor="single_feature_linear",
        memo="light smooth (w=5) + best single wavelength",
    ),
    "candidate_linear_idx616_raw": ExperimentSpec(
        preprocessor="spectral",
        predictor="single_feature_fixed_index",
        memo="raw index 616 fixed (Public 1f band)",
    ),
    "candidate_linear_idx616_center": ExperimentSpec(
        preprocessor="spectral_center",
        predictor="single_feature_fixed_index",
        memo="row center + index 616 fixed",
    ),
    "candidate_linear_idx616_blend25": ExperimentSpec(
        preprocessor="spectral_blend_snv25",
        predictor="single_feature_fixed_index",
        memo="raw/SNV blend25 + index 616 fixed",
    ),
    "candidate_linear_idx616_smooth5": ExperimentSpec(
        preprocessor="spectral_smooth5",
        predictor="single_feature_fixed_index",
        memo="smooth w=5 + index 616 fixed",
    ),
    "candidate_linear_1f_blend_snv25_nn_bias": ExperimentSpec(
        preprocessor="spectral_blend_snv25",
        predictor="nearest_train_species_bias_corrected",
        memo="raw75%+SNV25% + nearest-train-species residual correction",
    ),
    "group_curve_blend_spectral_meta": ExperimentSpec(
        preprocessor="spectral_meta",
        predictor="group_curve_blend",
        memo="raw spectral + group curve blend (meta columns)",
    ),
    "candidate_linear_1f_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="single_feature_linear",
        memo="best single wavelength linear regression + SNV",
    ),
    "candidate_linear_1f_diff1": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="single_feature_linear",
        memo="best single wavelength linear regression + first difference",
    ),
    "candidate_linear_1f_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_linear",
        memo="best single wavelength linear regression + SNV + first difference",
    ),
    "candidate_linear_1f_stable_wavelength_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_stable_wavelength",
        memo="stable top-1 wavelength across train chunks + SNV + first difference",
    ),
    "candidate_linear_1f_loo_consensus_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_loo_consensus_wavelength",
        memo="LOO species consensus top-1 wavelength + SNV + first difference",
    ),
    "candidate_linear_2f_stable_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="dual_stable_wavelength_ridge",
        memo="stable top-2 wavelengths + Ridge alpha=1000 + SNV + first difference",
    ),
    "candidate_linear_1f_loo_rmse_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_loo_rmse_wavelength",
        memo="top-30 candidates; pick wavelength minimizing LOO species RMSE + SNV + diff1",
    ),
    "candidate_linear_1f_rank2_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_second_wavelength",
        memo="2nd-best correlated wavelength linear regression + SNV + first difference",
    ),
    "candidate_linear_1f_snv_diff1_oof_intercept": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_linear_oof_intercept",
        memo="anchor wavelength + LOO species OOF mean intercept correction + SNV + diff1",
    ),
    "candidate_linear_top3_median_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_top3_median",
        memo="median of top-3 correlated wavelength linear models + SNV + first difference",
    ),
    "candidate_linear_blend_1f_2pct_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="dual_prep_anchor_raw_blend",
        memo="98% anchor snv+diff1 + 2% raw 1f (both Public 17.496 measured)",
    ),
    "candidate_linear_1f_nearest_train_species_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="nearest_train_species_linear",
        memo="test species mapped to spectrally nearest train species + per-species 1f linear",
    ),
    "candidate_linear_1f_snv_diff1_clip": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_linear_clipped",
        memo="best single wavelength linear regression + SNV + first difference + clipped",
    ),
    "candidate_linear_1f_snv_diff1_lowtail_mild": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_linear_low_tail_mild",
        memo="best single wavelength linear regression + SNV + first difference + mild low-tail extension",
    ),
    "candidate_linear_1f_snv_diff1_lowtail_strong": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="single_feature_linear_low_tail_strong",
        memo="best single wavelength linear regression + SNV + first difference + strong low-tail extension",
    ),
    "topk3_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk3_ridge",
        memo="top-3 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk5_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk5_ridge",
        memo="top-5 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk4_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk4_ridge",
        memo="top-4 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk6_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_ridge",
        memo="top-6 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk8_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk8_ridge",
        memo="top-8 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk5_ridge_a30_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk5_ridge_a30",
        memo="top-5 correlated wavelengths + Ridge alpha=30 + SNV + first difference",
    ),
    "topk5_ridge_a300_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk5_ridge_a300",
        memo="top-5 correlated wavelengths + Ridge alpha=300 + SNV + first difference",
    ),
    "topk5_stable_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk5_stable_ridge",
        memo="stable top-5 wavelengths + Ridge + SNV + first difference",
    ),
    "topk_ridge_cv_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk_ridge_cv",
        memo="top-5 wavelengths + RidgeCV + SNV + first difference",
    ),
    "topk_elastic_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk_elastic",
        memo="top-5 wavelengths + ElasticNet + SNV + first difference",
    ),
    "topk_ridge_ensemble_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk_ridge_ensemble",
        memo="ensemble of top-k Ridge variants + SNV + first difference",
    ),
    "topk6_weighted_q70_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_weighted_q70",
        memo="top-6 Ridge weighted for high-moisture samples + SNV + first difference",
    ),
    "topk6_weighted_q80_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_weighted_q80",
        memo="top-6 Ridge strongly weighted for high-moisture samples + SNV + first difference",
    ),
    "topk8_weighted_q70_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk8_weighted_q70",
        memo="top-8 Ridge weighted for high-moisture samples + SNV + first difference",
    ),
    "topk_quantile_blend_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk_quantile_blend",
        memo="top-6 quantile blend for upper range + SNV + first difference",
    ),
    "mono_topk6_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="meta_monotonic_topk6_ridge",
        memo="top-6 Ridge + SNV + first difference + species-wise monotonic correction",
    ),
    "mono_topk5_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="meta_monotonic_topk5_ridge",
        memo="top-5 Ridge + SNV + first difference + species-wise monotonic correction",
    ),
    "mono_quantile_blend_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="meta_monotonic_quantile_blend",
        memo="quantile blend + SNV + first difference + species-wise monotonic correction",
    ),
    "mono_topk5_ridge_snv_sg_diff1": ExperimentSpec(
        preprocessor="spectral_snv_sg_diff1_meta",
        predictor="meta_monotonic_topk5_ridge",
        memo="top-5 Ridge + SNV + 5-point smooth + monotonic correction",
    ),
    "group_curve_quantile_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_template_quantile",
        memo="group-level drying curve template + quantile base + SNV + first difference",
    ),
    "group_curve_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_template_ridge",
        memo="group-level drying curve template + ridge base + SNV + first difference",
    ),
    "group_curve_wide_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_template_wide",
        memo="wide-range group drying curve template + quantile base + SNV + first difference",
    ),
    "group_curve_residual_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_residual",
        memo="group drying curve + spectral residual correction + SNV + first difference",
    ),
    "group_curve_upper_anchor_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_upper_anchor",
        memo="group drying curve with upper-anchor range lift + quantile base + SNV + first difference",
    ),
    "group_curve_range_cal_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_range_calibrated",
        memo="range-calibrated group drying curve + ridge base + SNV + first difference",
    ),
    "group_curve_blend_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="group_curve_blend",
        memo="blend of ridge and quantile group drying curves + SNV + first difference",
    ),
    "topk12_hgbdt_snv_diff1_meta": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="topk12_hgbdt",
        memo="top-12 wavelength HistGradientBoosting + monotonic group projection + SNV + first difference",
    ),
    "topk24_hgbdt_snv_diff1_meta": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="topk24_hgbdt",
        memo="top-24 wavelength HistGradientBoosting + monotonic group projection + SNV + first difference",
    ),
    "pls4_hgbdt_snv_diff1_meta": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="pls4_hgbdt",
        memo="PLS-4 HistGradientBoosting + monotonic group projection + SNV + first difference",
    ),
    "pls6_hgbdt_snv_diff1_meta": ExperimentSpec(
        preprocessor="spectral_snv_diff1_meta",
        predictor="pls6_hgbdt",
        memo="PLS-6 HistGradientBoosting + monotonic group projection + SNV + first difference",
    ),
    "topk6_upper_cal_q70_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_upper_cal_q70",
        memo="top-6 Ridge upper-range post calibration q70 + SNV + first difference",
    ),
    "topk6_upper_cal_q80_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_upper_cal_q80",
        memo="top-6 Ridge upper-range post calibration q80 + SNV + first difference",
    ),
    "topk6_mean_shift_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk6_mean_shift",
        memo="top-6 Ridge mean-based post calibration + SNV + first difference",
    ),
    "topk12_ridge_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk12_ridge",
        memo="top-12 correlated wavelengths + Ridge + SNV + first difference",
    ),
    "topk_huber_snv_diff1": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="topk_huber",
        memo="top correlated wavelengths + Huber + SNV + first difference",
    ),
    "topk3_ridge_diff1": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="topk3_ridge",
        memo="top-3 correlated wavelengths + Ridge + first difference",
    ),
    "topk5_ridge_diff1": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="topk5_ridge",
        memo="top-5 correlated wavelengths + Ridge + first difference",
    ),
    "topk_huber_diff1": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="topk_huber",
        memo="top correlated wavelengths + Huber + first difference",
    ),
    "topk5_ridge_snv_diff2": ExperimentSpec(
        preprocessor="spectral_snv_diff2",
        predictor="topk5_ridge",
        memo="top-5 correlated wavelengths + Ridge + SNV + second difference",
    ),
    "topk5_ridge_snv_sg_diff1": ExperimentSpec(
        preprocessor="spectral_snv_sg_diff1",
        predictor="topk5_ridge",
        memo="top-5 correlated wavelengths + Ridge + SNV + 5-point smooth + first difference",
    ),
    "topk5_ridge_snv_area_diff1": ExperimentSpec(
        preprocessor="spectral_snv_area_diff1",
        predictor="topk5_ridge",
        memo="top-5 correlated wavelengths + Ridge + SNV + area normalization + first difference",
    ),
    "topk_huber_snv_sg_diff1": ExperimentSpec(
        preprocessor="spectral_snv_sg_diff1",
        predictor="topk_huber",
        memo="top correlated wavelengths + Huber + SNV + 5-point smooth + first difference",
    ),
    "candidate_linear_topk": ExperimentSpec(
        preprocessor="spectral",
        predictor="topk_sgd_linear",
        memo="candidate2: top-k correlated wavelengths + SGD linear",
    ),
    "candidate_linear_topk_clip": ExperimentSpec(
        preprocessor="spectral",
        predictor="topk_sgd_linear_clipped",
        memo="candidate3: top-k correlated wavelengths + SGD linear (clipped)",
    ),
    "pls_snv_c8": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + SNV",
    ),
    "pls_snv_c2": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pls_c2",
        memo="PLSRegression n_components=2 + SNV",
    ),
    "pls_snv_c4": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pls_c4",
        memo="PLSRegression n_components=4 + SNV",
    ),
    "pls_snv_c6": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pls_c6",
        memo="PLSRegression n_components=6 + SNV",
    ),
    "pls_snv_c12": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pls_c12",
        memo="PLSRegression n_components=12 + SNV",
    ),
    "pls_diff1_c8": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + first difference",
    ),
    "pls_snv_diff1_c8": ExperimentSpec(
        preprocessor="spectral_snv_diff1",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + SNV + first difference",
    ),
    "pls_smooth_diff1_c4": ExperimentSpec(
        preprocessor="spectral_smooth_diff1",
        predictor="pls_c4",
        memo="PLSRegression n_components=4 + smooth + first difference",
    ),
    "pls_smooth_diff1_c6": ExperimentSpec(
        preprocessor="spectral_smooth_diff1",
        predictor="pls_c6",
        memo="PLSRegression n_components=6 + smooth + first difference",
    ),
    "pls_smooth_diff1_c8": ExperimentSpec(
        preprocessor="spectral_smooth_diff1",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + smooth + first difference",
    ),
    "pls_snv_smooth_diff1_c4": ExperimentSpec(
        preprocessor="spectral_snv_smooth_diff1",
        predictor="pls_c4",
        memo="PLSRegression n_components=4 + SNV + smooth + first difference",
    ),
    "pls_snv_smooth_diff1_c6": ExperimentSpec(
        preprocessor="spectral_snv_smooth_diff1",
        predictor="pls_c6",
        memo="PLSRegression n_components=6 + SNV + smooth + first difference",
    ),
    "pls_snv_smooth_diff1_c8": ExperimentSpec(
        preprocessor="spectral_snv_smooth_diff1",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + SNV + smooth + first difference",
    ),
    "pls_msc_c4": ExperimentSpec(
        preprocessor="spectral_msc",
        predictor="pls_c4",
        memo="PLSRegression n_components=4 + MSC",
    ),
    "pls_msc_c6": ExperimentSpec(
        preprocessor="spectral_msc",
        predictor="pls_c6",
        memo="PLSRegression n_components=6 + MSC",
    ),
    "pls_msc_c8": ExperimentSpec(
        preprocessor="spectral_msc",
        predictor="pls_c8",
        memo="PLSRegression n_components=8 + MSC",
    ),
    "ridge_snv_alpha10": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="ridge_alpha10",
        memo="Ridge alpha=10 + SNV",
    ),
    "ridge_diff1_alpha10": ExperimentSpec(
        preprocessor="spectral_diff1",
        predictor="ridge_alpha10",
        memo="Ridge alpha=10 + first difference",
    ),
    "ridge_snv_alpha100": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="ridge_alpha100",
        memo="Ridge alpha=100 + SNV",
    ),
    "ridge_snv_alpha1": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="ridge_alpha1",
        memo="Ridge alpha=1 + SNV",
    ),
    "ridge_snv_alpha1000": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="ridge_alpha1000",
        memo="Ridge alpha=1000 + SNV",
    ),
    "log_ridge_snv_alpha100": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="log_ridge_alpha100",
        memo="log1p target + Ridge alpha=100 + SNV",
    ),
    "bayesian_ridge_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="bayesian_ridge",
        memo="BayesianRidge + SNV",
    ),
    "huber_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="huber",
        memo="HuberRegressor + SNV",
    ),
    "pca50_ridge_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="pca50_ridge_alpha100",
        memo="PCA(50) + Ridge alpha=100 + SNV",
    ),
    "svr_rbf_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="svr_rbf_c10",
        memo="RBF SVR + SNV",
    ),
    "svr_linear_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="svr_linear_c1",
        memo="LinearSVR + SNV",
    ),
    "elastic_snv": ExperimentSpec(
        preprocessor="spectral_snv",
        predictor="elastic_net",
        memo="ElasticNet + SNV",
    ),
}


def resolve_experiment(name: str) -> tuple[Preprocessor, Predictor, str]:
    if name not in EXPERIMENTS:
        raise KeyError(f"未知の experiment: {name}. 利用可能: {list(EXPERIMENTS)}")
    return _build(EXPERIMENTS[name])
