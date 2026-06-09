CANDIDATE_MODEL_SET_20 = [
    "Sub_IForest", "POLY", "MatrixProfile", "KShapeAD", "SAND",
    "Sub_PCA", "CNN", "Sub_HBOS", "TranAD", "OmniAnomaly",
    "USAD", "TimesFM",
    "Sub_LOF", "SR", "Sub_KNN", "Sub_OCSVM", "KMeansAD_U",
    "AutoEncoder", "TimesNet", "AnomalyTransformer",
]

CATCH22_NAMES = [
    'DN_HistogramMode_5', 'DN_HistogramMode_10', 'CO_f1ecac', 'CO_FirstMin_ac',
    'CO_HistogramAMI_even_2_5', 'CO_trev_1_num', 'MD_hrv_classic_pnn40',
    'SB_BinaryStats_mean_longstretch1', 'SB_TransitionMatrix_3ac_sumdiagcov',
    'PD_PeriodicityWang_th0_01', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
    'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'FC_LocalSimple_mean1_tauresrat',
    'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd',
    'SP_Summaries_welch_rect_area_5_1', 'SB_BinaryStats_diff_longstretch0',
    'SB_MotifThree_quantile_hh', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
    'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SP_Summaries_welch_rect_centroid',
    'FC_LocalSimple_mean3_stderr'
]

DOMAINS = [
    "ID", "WebService", "Medical", "Facility", "Synthetic",
    "HumanActivity", "Sensor", "Environment", "Finance", "Traffic",
]

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_MIN_WINDOWS = 100
DEFAULT_GAP_THRESHOLD = 0.05
DEFAULT_RRF_KAPPA = 60
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = 10
DEFAULT_LOOCV_ALPHA = 1.0
DEFAULT_BACKGROUND_SAMPLE_SIZE = 200
DEFAULT_EVAL_SLIDING_WINDOW = 100
