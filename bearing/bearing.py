import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from typing import Dict, List, Tuple, Optional


class VibrationFaultDetector:
    """
    滚动轴承与转子振动故障检测器
    功能：原始频谱识别转子故障(不平衡、不对中、松动)；包络谱识别轴承故障(内圈、外圈、滚动体)
    """
    def __init__(
        self,
        fs: float,
        hp_cutoff: float = 800.0,
        lp_cutoff_env: float = 1000.0,
        harmonic_max_k: int = 3,   # 改：只匹配1~3阶谐波，降低串扰
        peak_min_distance: int = 3,
        prom_ratio: float = 0.15,
        tol_ratio: float = 0.05
    ):
        """
        :param fs: 采样率 Hz
        :param hp_cutoff: 包络解调高通截止频率
        :param lp_cutoff_env: 包络解调低通截止频率
        :param harmonic_max_k: 谐波匹配最大阶数
        :param peak_min_distance: 频谱峰值最小间隔(点数)，抑制密集伪峰
        :param prom_ratio: 峰值突出度最低占比阈值，相对于最大突出度
        :param tol_ratio: 特征频率匹配容差比例（相对于转频fr）
        """
        self.fs = fs
        self.hp_cutoff = hp_cutoff
        self.lp_cutoff_env = lp_cutoff_env # <---- 修复这里变量名
        self.harmonic_max_k = harmonic_max_k
        self.peak_min_distance = peak_min_distance
        self.prom_ratio = prom_ratio
        self.tol_ratio = tol_ratio

    @staticmethod
    def median(data: np.ndarray) -> float:
        return float(np.median(data))

    @staticmethod
    def _validate_signal(signal: np.ndarray) -> bool:
        """输入信号合法性校验"""
        if not isinstance(signal, np.ndarray):
            return False
        if signal.ndim != 1:
            return False
        if len(signal) < 32:
            return False
        return True

    def bandpass_filter(self, signal: np.ndarray, low: float, high: float) -> np.ndarray:
        """4阶巴特沃斯带通滤波"""
        if not self._validate_signal(signal):
            raise ValueError("Invalid input signal for bandpass filter")
        b, a = butter(4, [low, high], btype="bandpass", fs=self.fs)
        return filtfilt(b, a, signal)

    def highpass_filter(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """4阶巴特沃斯高通滤波"""
        if not self._validate_signal(signal):
            raise ValueError("Invalid input signal for highpass filter")
        b, a = butter(4, cutoff, btype="highpass", fs=self.fs)
        return filtfilt(b, a, signal)

    def lowpass_filter(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """4阶巴特沃斯低通滤波"""
        if not self._validate_signal(signal):
            raise ValueError("Invalid input signal for lowpass filter")
        b, a = butter(4, cutoff, btype="lowpass", fs=self.fs)
        return filtfilt(b, a, signal)

    def envelope_demodulation(self, signal: np.ndarray) -> np.ndarray:
        """
        希尔伯特变换包络解调，用于提取轴承冲击调制信号
        增加汉宁窗抑制希尔伯特变换边界效应
        """
        if not self._validate_signal(signal):
            raise ValueError("Invalid input signal for envelope demodulation")
        n = len(signal)
        window = np.hanning(n)
        sig_win = signal * window

        sig_hp = self.highpass_filter(sig_win, self.hp_cutoff)
        analytic = hilbert(sig_hp)
        env = np.abs(analytic)
        env = self.lowpass_filter(env, self.lp_cutoff_env)
        return env

    def fft_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单边FFT频谱，修正幅值缩放
        return: freq array, amp array
        """
        if not self._validate_signal(signal):
            raise ValueError("Invalid input signal for FFT")
        n = len(signal)
        fft_vals = np.fft.fft(signal)
        amp = np.abs(fft_vals)
        freq = np.fft.fftfreq(n, 1.0 / self.fs)

        mask = freq >= 0
        freq = freq[mask]
        amp = amp[mask]

        # 单边谱幅值修正：DC除以n，其余除以n/2
        amp[0] = amp[0] / n
        if len(amp) > 1:
            amp[1:] = amp[1:] / (n / 2.0)
        return freq, amp

    def find_spectrum_peaks(self, freq: np.ndarray, amp: np.ndarray) -> List[Dict]:
        """
        频谱峰值自动检索，动态噪声基底
        return list: [{"freq": float, "amp": float, "prominence": float}, ...]
        """
        n = len(amp)
        if n < 2:
            return []

        # 改进噪声基底：取频谱后20%中位数作为噪声底
        noise_window = amp[int(n * 0.8):]
        noise_floor = self.median(noise_window)
        min_height = noise_floor * 2.0
        min_prom = noise_floor * 1.5

        peaks_idx, props = find_peaks(
            amp,
            height=min_height,
            prominence=min_prom,
            distance=self.peak_min_distance
        )
        peak_list = []
        for i, peak_idx in enumerate(peaks_idx):
            peak_list.append({
                "freq": float(freq[peak_idx]),
                "amp": float(amp[peak_idx]),
                "prominence": float(props["prominences"][i]),
                "noise_floor": float(noise_floor)
            })
        return peak_list

    def match_harmonic_series(self, peaks: List[Dict], f_target: float, tol: float) -> Tuple[int, float]:
        """
        谐波序列匹配 f_target*1, f_target*2 ... f_target*max_k
        仅匹配突出度足够的峰值，过滤噪声伪峰
        :return hit_count, score(0~1)
        """
        if not peaks:
            return 0, 0.0
        max_prom = max(p["prominence"] for p in peaks)
        min_prom_threshold = max_prom * self.prom_ratio

        hit = 0
        for k in range(1, self.harmonic_max_k + 1):
            ft = f_target * k
            for p in peaks:
                if abs(p["freq"] - ft) <= tol and p["prominence"] >= min_prom_threshold:
                    hit += 1
                    break
        score = hit / self.harmonic_max_k
        return hit, score

    def match_sideband(self, peaks: List[Dict], center_f: float, side_step: float, tol: float) -> Tuple[int, float]:
        """
        边带匹配：中心频率 ±1*side_step, ±2*side_step
        仅匹配突出度足够的峰值
        :return hit_count, score(0~1)
        """
        if not peaks:
            return 0, 0.0
        max_prom = max(p["prominence"] for p in peaks)
        min_prom_threshold = max_prom * self.prom_ratio

        hit = 0
        for n in range(1, 3):
            fl = center_f - n * side_step
            fr = center_f + n * side_step
            has_left = False
            has_right = False
            for p in peaks:
                if abs(p["freq"] - fl) <= tol and p["prominence"] >= min_prom_threshold:
                    has_left = True
                if abs(p["freq"] - fr) <= tol and p["prominence"] >= min_prom_threshold:
                    has_right = True
            if has_left and has_right:
                hit += 1
        score = hit / 2.0
        return hit, score

    def detect(
            self,
            wave: np.ndarray,
            rpm: float,
            bpfo: float,
            bpfi: float,
            bsf: float,
            ftf: float
    ) -> Dict:
        """
        振动故障检测主入口
        :param wave: 原始加速度时域波形一维数组
        :param rpm: 转轴转速 (r/min)
        :param bpfo: 轴承外圈特征频率
        :param bpfi: 轴承内圈特征频率
        :param bsf: 滚动体特征频率
        :param ftf: 保持架特征频率
        :return: 诊断结果字典，包含转频、各故障分数、判定结论、峰值列表
        """
        # 输入校验
        if not self._validate_signal(wave):
            raise ValueError("wave must be 1D numpy array and length >=32")
        if rpm <= 0:
            raise ValueError("rpm must > 0")

        # 预处理：去除直流分量
        wave = wave - np.mean(wave)
        fr = rpm / 60.0
        tol = fr * self.tol_ratio

        # ========== 1. 原始频谱：转子故障（不平衡，不对中，松动） ==========
        freq_full, amp_full = self.fft_spectrum(wave)
        peaks_full = self.find_spectrum_peaks(freq_full, amp_full)

        amp1, amp2, amp3 = 0.0, 0.0, 0.0
        noise_floor_full = 1e-9
        for p in peaks_full:
            noise_floor_full = p["noise_floor"]
            if abs(p["freq"] - fr) < tol:
                amp1 = p["amp"]
            if abs(p["freq"] - 2 * fr) < tol:
                amp2 = p["amp"]
            if abs(p["freq"] - 3 * fr) < tol:
                amp3 = p["amp"]

        # ========== 【修复转子打分：使用信噪比，不再单纯占比】 ==========
        snr1 = amp1 / noise_floor_full
        snr2 = amp2 / noise_floor_full
        # 把信噪比压缩到0~1区间，饱和上限50倍噪声
        score_unbalance = np.clip(snr1 / 50, 0, 1.0)
        score_misalignment = np.clip(snr2 / 50, 0, 1.0)
        _, score_looseness = self.match_harmonic_series(peaks_full, fr, tol)

        # ========== 2. 包络谱：轴承故障特征提取 ==========
        env_signal = self.envelope_demodulation(wave)
        freq_env, amp_env = self.fft_spectrum(env_signal)
        peaks_env = self.find_spectrum_peaks(freq_env, amp_env)

        # 外圈故障 BPFO
        _, score_bpfo = self.match_harmonic_series(peaks_env, bpfo, tol)
        # 内圈故障 BPFI：谐波 + 转频边带
        _, score_bpfi_harm = self.match_harmonic_series(peaks_env, bpfi, tol)
        _, score_bpfi_side = self.match_sideband(peaks_env, bpfi, fr, tol)
        score_inner = (score_bpfi_harm + score_bpfi_side) / 2.0
        # 滚动体 BSF：谐波 + FTF边带
        _, score_bsf_harm = self.match_harmonic_series(peaks_env, bsf, tol)
        _, score_bsf_side = self.match_sideband(peaks_env, bsf, ftf, tol)
        score_rolling = (score_bsf_harm + score_bsf_side) / 2.0

        # ========== 3. 分组打分：转子组 / 轴承组 竞争互斥 ==========
        rotor_scores = {
            "unbalance": score_unbalance,
            "misalignment": score_misalignment,
            "looseness": score_looseness
        }
        bearing_scores = {
            "outer_race": score_bpfo,
            "inner_race": score_inner,
            "rolling_element": score_rolling
        }
        all_scores = {**rotor_scores, **bearing_scores}

        max_rotor_key = max(rotor_scores, key=rotor_scores.get)
        max_rotor_score = rotor_scores[max_rotor_key]
        max_bearing_key = max(bearing_scores, key=bearing_scores.get)
        max_bearing_score = bearing_scores[max_bearing_key]

        # 组间竞争：优先取高分组
        if max_bearing_score > max_rotor_score:
            max_key = max_bearing_key
            max_score = max_bearing_score
        else:
            max_key = max_rotor_key
            max_score = max_rotor_score

        # 故障判定
        if max_score < 0.25:
            fault_result = "Healthy"
        elif max_score < 0.6:
            fault_result = f"Suspected {max_key}"
        else:
            fault_result = f"Fault: {max_key}"

        return {
            "fr": fr,
            "tol": tol,
            "scores": all_scores,
            "max_rotor": {"name": max_rotor_key, "score": max_rotor_score},
            "max_bearing": {"name": max_bearing_key, "score": max_bearing_score},
            "fault_result": fault_result,
            "max_confidence": max_score,
            "peaks_full": peaks_full,
            "peaks_env": peaks_env
        }


# ====================== Demo 测试入口 ======================
if __name__ == "__main__":
    fs = 10000  # 采样率 Hz
    rpm = 1480
    fr = rpm / 60
    # 6205轴承特征频率
    BPFO = 89
    BPFI = 133
    BSF = 57
    FTF = 9.9

    # 构造模拟信号：内圈故障，带转频调制边带
    t_total = 2
    t = np.linspace(0, t_total, int(fs * t_total), endpoint=False)
    wave = np.zeros_like(t)
    # 减小独立转频分量，避免不平衡分数直接饱和
    wave += 0.02 * np.sin(2 * np.pi * fr * t)
    modulator = 1 + 0.3 * np.sin(2 * np.pi * fr * t)
    wave += modulator * 0.8 * np.sin(2 * np.pi * BPFI * t)
    wave += 0.15 * np.random.randn(len(t))

    # 实例化检测器，缩小容差减少谐波串扰
    detector = VibrationFaultDetector(
        fs=fs,
        hp_cutoff=800,
        lp_cutoff_env=1000,
        harmonic_max_k=3,
        peak_min_distance=3,
        prom_ratio=0.15,
        tol_ratio=0.03
    )
    res = detector.detect(wave, rpm, BPFO, BPFI, BSF, FTF)

    print("===== 故障识别结果 =====")
    print(f"转轴转频 fr={res['fr']:.2f} Hz")
    print(f"频率匹配容差 tol={res['tol']:.3f} Hz")
    print(f"转子组最高：{res['max_rotor']['name']:12s} {res['max_rotor']['score']:.3f}")
    print(f"轴承组最高：{res['max_bearing']['name']:12s} {res['max_bearing']['score']:.3f}")
    print("\n各故障置信度：")
    for name, val in res["scores"].items():
        print(f"  {name:16s}: {val:.3f}")
    print(f"\n判定结果：{res['fault_result']}")
    print(f"最高置信：{res['max_confidence']:.3f}")
