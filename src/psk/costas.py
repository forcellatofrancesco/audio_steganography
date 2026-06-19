# psk/costas.py

import numpy as np
from util.types import value_array

try:
    import cupy as cp
    _CUPY = True
except ImportError:
    _CUPY = False

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator


@njit(cache=True)
def _costas_kernel_numba(sig, freq_init, K1, K2, modulation_bpsk):
    N = sig.shape[0]
    phase_track = np.zeros(N, dtype=np.float64)
    phase = 0.0
    freq = freq_init
    integrator = 0.0

    for n in range(N):
        i_bb = sig[n] * np.cos(phase)
        q_bb = sig[n] * (-np.sin(phase))

        if modulation_bpsk:
            err = i_bb * q_bb
        else:
            si = 1.0 if i_bb >= 0.0 else -1.0
            sq = 1.0 if q_bb >= 0.0 else -1.0
            err = si * q_bb - sq * i_bb

        integrator += K2 * err
        freq += K1 * err + integrator
        phase += freq
        while phase > np.pi:
            phase -= 2 * np.pi
        while phase < -np.pi:
            phase += 2 * np.pi

        phase_track[n] = phase

    return phase_track


def _costas_cupy_blockwise(
    sig_gpu: "cp.ndarray",
    freq_init: float,
    K1: float,
    K2: float,
    modulation_bpsk: bool,
    block_size: int,
) -> "cp.ndarray":
    """
    Costas loop ibrida CPU/GPU:
    - Dentro ogni blocco: mixer + phase detector vettorizzati su GPU (CuPy)
    - Tra blocchi: aggiornamento di fase/frequenza sulla CPU (scalare)

    Con block_size grande (es. 1s = 44100 campioni) il trasferimento di
    stato CPU↔GPU è trascurabile rispetto al lavoro GPU per blocco.
    """
    N = int(sig_gpu.size)
    phase_track_gpu = cp.empty(N, dtype=cp.float64)

    phase = 0.0
    freq = freq_init
    integrator = 0.0

    n = 0
    while n < N:
        blk_end = min(n + block_size, N)
        blk = sig_gpu[n:blk_end]
        L = int(blk.size)

        # Fase locale: vettorizzata su GPU
        local_t = cp.arange(L, dtype=cp.float64)
        local_phase = phase + freq * local_t
        local_phase = local_phase - 2 * cp.pi * cp.floor(
            (local_phase + cp.pi) / (2 * cp.pi)
        )
        phase_track_gpu[n:blk_end] = local_phase

        # Mixer su GPU
        i_bb = blk * cp.cos(local_phase)
        q_bb = blk * (-cp.sin(local_phase))

        # Phase error: riduzione GPU → scalare CPU
        if modulation_bpsk:
            err = float(cp.mean(i_bb * q_bb))
        else:
            err = float(cp.mean(cp.sign(i_bb) * q_bb - cp.sign(q_bb) * i_bb))

        # Aggiorna stato (CPU, scalare)
        integrator += K2 * err
        freq += K1 * err + integrator
        phase = float(local_phase[-1]) + freq
        phase = (phase + np.pi) % (2 * np.pi) - np.pi

        n = blk_end

    return phase_track_gpu


def _compute_loop_gains(
    loop_bw: float,
    sample_rate: int,
    zeta: float = 0.707,
    K0: float = 1.0,
    Kd: float = 0.5,
) -> tuple[float, float]:
    Bn = loop_bw * sample_rate
    denom = 1 + 2 * zeta * Bn / (Kd * K0) + (Bn / (Kd * K0)) ** 2
    K1 = (4 * zeta * (Bn / (Kd * K0))) / denom
    K2 = (4 * (Bn / (Kd * K0)) ** 2) / denom
    return K1, K2


def warmup_costas_loop() -> None:
    if _CUPY:
        # Warmup CuPy: prima chiamata compila i kernel CUDA
        dummy_gpu = cp.zeros(256, dtype=cp.float64)
        K1, K2 = _compute_loop_gains(0.01, 44100)
        _costas_cupy_blockwise(dummy_gpu, 0.1, K1, K2, True, 256)
        print("[costas] CuPy kernel warmup completato")
    elif _NUMBA:
        dummy = np.zeros(256, dtype=np.float64)
        K1, K2 = _compute_loop_gains(0.01, 44100)
        _costas_kernel_numba(dummy, 0.1, K1, K2, True)
        print("[costas] Numba kernel warmup completato")
    else:
        print("[costas] fallback NumPy a blocchi (no CuPy, no Numba)")


def run_costas_loop(
    waveform: value_array,
    frequency: int,
    sample_rate: int,
    loop_bw: float = 0.015,
    zeta: float = 0.707,
    modulation: str = "bpsk",
) -> np.ndarray:
    """
    Restituisce sempre un np.ndarray (CPU), indipendentemente dal backend.
    """
    sig = np.asarray(waveform, dtype=np.float64)
    if sig.size == 0:
        return np.zeros(0, dtype=np.float64)

    K1, K2 = _compute_loop_gains(loop_bw, sample_rate, zeta)
    freq_init = 2 * np.pi * frequency / sample_rate
    modulation_bpsk = modulation == "bpsk"

    duration = sig.size / sample_rate
    backend = "cupy" if _CUPY else ("numba" if _NUMBA else "numpy")
    print(f"[costas] {sig.size} campioni ({duration:.1f}s), backend={backend}")

    if _CUPY:
        # Block size grande: 1 secondo intero per massimizzare il lavoro GPU
        block_size = sample_rate
        sig_gpu = cp.asarray(sig)
        phase_track_gpu = _costas_cupy_blockwise(
            sig_gpu, freq_init, K1, K2, modulation_bpsk, block_size
        )
        result = cp.asnumpy(phase_track_gpu)
    elif _NUMBA:
        result = _costas_kernel_numba(sig, freq_init, K1, K2, modulation_bpsk)
    else:
        block_size = max(1, sample_rate // 10)
        phase = 0.0
        freq = freq_init
        integrator = 0.0
        result = np.empty(sig.size, dtype=np.float64)
        n = 0
        while n < sig.size:
            blk_end = min(n + block_size, sig.size)
            blk = sig[n:blk_end]
            L = blk.size
            local_t = np.arange(L, dtype=np.float64)
            local_phase = phase + freq * local_t
            local_phase -= 2 * np.pi * np.floor(
                (local_phase + np.pi) / (2 * np.pi)
            )
            result[n:blk_end] = local_phase
            i_bb = blk * np.cos(local_phase)
            q_bb = blk * (-np.sin(local_phase))
            if modulation_bpsk:
                err = float(np.mean(i_bb * q_bb))
            else:
                err = float(np.mean(np.sign(i_bb) * q_bb - np.sign(q_bb) * i_bb))
            integrator += K2 * err
            freq += K1 * err + integrator
            phase = float(local_phase[-1]) + freq
            phase = (phase + np.pi) % (2 * np.pi) - np.pi
            n = blk_end

    print("[costas] completato")
    return result