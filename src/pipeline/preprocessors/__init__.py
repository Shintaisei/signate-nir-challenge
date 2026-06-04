from pipeline.preprocessors.spectral import SpectralPreprocessor
from pipeline.preprocessors.spectral_transforms import (
    SpectralAreaPreprocessor,
    SpectralBlendSnv25Preprocessor,
    SpectralBlendSnv50Preprocessor,
    SpectralCenterPreprocessor,
    SpectralDetrendPreprocessor,
    SpectralDiff1Preprocessor,
    SpectralL2NormPreprocessor,
    SpectralDiff2Preprocessor,
    SpectralMscPreprocessor,
    SpectralMultiviewPreprocessor,
    SpectralSgDiff1Preprocessor,
    SpectralSmoothDiff1Preprocessor,
    SpectralSmooth3Preprocessor,
    SpectralSmooth5Preprocessor,
    SpectralSmoothPreprocessor,
    SpectralSnvAreaDiff1Preprocessor,
    SpectralSnvDiff1Preprocessor,
    SpectralMetaPreprocessor,
    SpectralSnvDiff1MetaPreprocessor,
    SpectralSnvDiff2Preprocessor,
    SpectralSnvSgDiff1Preprocessor,
    SpectralSnvSgDiff1MetaPreprocessor,
    SpectralSnvSmoothDiff1Preprocessor,
    SpectralSnvPreprocessor,
)

PREPROCESSOR_REGISTRY: dict[str, type] = {
    "spectral": SpectralPreprocessor,
    "spectral_snv": SpectralSnvPreprocessor,
    "spectral_diff1": SpectralDiff1Preprocessor,
    "spectral_diff2": SpectralDiff2Preprocessor,
    "spectral_snv_diff1": SpectralSnvDiff1Preprocessor,
    "spectral_meta": SpectralMetaPreprocessor,
    "spectral_snv_diff1_meta": SpectralSnvDiff1MetaPreprocessor,
    "spectral_snv_diff2": SpectralSnvDiff2Preprocessor,
    "spectral_center": SpectralCenterPreprocessor,
    "spectral_l2norm": SpectralL2NormPreprocessor,
    "spectral_blend_snv25": SpectralBlendSnv25Preprocessor,
    "spectral_blend_snv50": SpectralBlendSnv50Preprocessor,
    "spectral_smooth3": SpectralSmooth3Preprocessor,
    "spectral_smooth5": SpectralSmooth5Preprocessor,
    "spectral_smooth": SpectralSmoothPreprocessor,
    "spectral_smooth_diff1": SpectralSmoothDiff1Preprocessor,
    "spectral_snv_smooth_diff1": SpectralSnvSmoothDiff1Preprocessor,
    "spectral_sg_diff1": SpectralSgDiff1Preprocessor,
    "spectral_snv_sg_diff1": SpectralSnvSgDiff1Preprocessor,
    "spectral_snv_sg_diff1_meta": SpectralSnvSgDiff1MetaPreprocessor,
    "spectral_area": SpectralAreaPreprocessor,
    "spectral_snv_area_diff1": SpectralSnvAreaDiff1Preprocessor,
    "spectral_detrend": SpectralDetrendPreprocessor,
    "spectral_msc": SpectralMscPreprocessor,
    "spectral_multiview": SpectralMultiviewPreprocessor,
}


def get_preprocessor(name: str):
    if name not in PREPROCESSOR_REGISTRY:
        raise KeyError(f"未知の preprocessor: {name}. 利用可能: {list(PREPROCESSOR_REGISTRY)}")
    return PREPROCESSOR_REGISTRY[name]()
