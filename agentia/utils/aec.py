import threading


class EchoCanceller:
    """Acoustic echo canceller using frequency-domain adaptive filtering (FDAF/NLMS).

    Removes speaker output (reference) from microphone input to prevent echo feedback.
    Handles sample rate mismatch between input (16kHz) and output (24kHz) automatically.
    """

    def __init__(
        self,
        block_size: int = 256,
        num_blocks: int = 12,
        mu: float = 0.5,
        delta: float = 1e-2,
        input_rate: int = 16000,
        output_rate: int = 24000,
    ):
        import numpy as np

        self._np = np
        N = block_size
        M = num_blocks
        F = N + 1  # rfft output length for 2N-point FFT

        self._N = N
        self._M = M
        self._mu = mu
        self._delta = delta
        self._input_rate = input_rate
        self._output_rate = output_rate

        # Adaptive filter taps in frequency domain
        self._W = np.zeros((M, F), dtype=np.complex128)
        # Reference delay line in frequency domain
        self._X_buf = np.zeros((M, F), dtype=np.complex128)
        # Running power estimate for NLMS normalisation
        self._P = np.ones(F, dtype=np.float64) * delta

        # Overlap buffers (overlap-save needs previous block)
        self._ref_overlap = np.zeros(N, dtype=np.float64)
        self._mic_overlap = np.zeros(N, dtype=np.float64)

        # Queues for incomplete blocks
        self._ref_queue = np.array([], dtype=np.float64)
        self._mic_pending = np.array([], dtype=np.float64)
        self._out_pending = np.array([], dtype=np.float64)

        self._lock = threading.Lock()

    def _resample(self, data: "numpy.ndarray", from_rate: int, to_rate: int):  # type: ignore[name-defined]
        if from_rate == to_rate:
            return data
        np = self._np
        n_out = int(len(data) * to_rate / from_rate)
        x_old = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, n_out)
        return np.interp(x_new, x_old, data)

    def feed_reference(self, data: bytes) -> None:
        """Feed playback audio as the reference signal. Called by AudioOutput."""
        np = self._np
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64) / 32768.0
        if self._output_rate != self._input_rate:
            samples = self._resample(samples, self._output_rate, self._input_rate)
        with self._lock:
            self._ref_queue = np.concatenate([self._ref_queue, samples])

    def process(self, mic_data: bytes) -> bytes:
        """Remove echo from microphone audio. Called by AudioInput."""
        np = self._np
        N = self._N
        mic = np.frombuffer(mic_data, dtype=np.int16).astype(np.float64) / 32768.0
        n_input = len(mic)

        with self._lock:
            self._mic_pending = np.concatenate([self._mic_pending, mic])
            output_blocks: list = []

            while len(self._mic_pending) >= N:
                mic_block = self._mic_pending[:N]
                self._mic_pending = self._mic_pending[N:]

                # Get matching reference block (zero if not available yet)
                if len(self._ref_queue) >= N:
                    ref_block = self._ref_queue[:N]
                    self._ref_queue = self._ref_queue[N:]
                else:
                    ref_block = np.zeros(N, dtype=np.float64)
                    if len(self._ref_queue) > 0:
                        ref_block[: len(self._ref_queue)] = self._ref_queue
                        self._ref_queue = np.array([], dtype=np.float64)

                output_blocks.append(self._process_block(mic_block, ref_block))

            if output_blocks:
                self._out_pending = np.concatenate([self._out_pending, *output_blocks])

            # Return exactly the same number of samples as input
            if len(self._out_pending) >= n_input:
                result = self._out_pending[:n_input]
                self._out_pending = self._out_pending[n_input:]
            else:
                # Not enough processed samples yet – pass through raw input
                return mic_data

        result = np.clip(result * 32768.0, -32768, 32767).astype(np.int16)
        return result.tobytes()

    def _process_block(self, mic_block, ref_block):
        np = self._np
        N = self._N

        # Overlap-save: build 2N segments from previous + current block
        ref_seg = np.concatenate([self._ref_overlap, ref_block])
        mic_seg = np.concatenate([self._mic_overlap, mic_block])
        self._ref_overlap = ref_block.copy()
        self._mic_overlap = mic_block.copy()

        # Forward FFT
        X = np.fft.rfft(ref_seg)
        D = np.fft.rfft(mic_seg)

        # Shift reference delay line and insert newest block
        self._X_buf = np.roll(self._X_buf, 1, axis=0)
        self._X_buf[0] = X

        # Echo estimate: sum of filter * reference across all taps
        Y = np.sum(self._W * self._X_buf, axis=0)

        # Error = desired (mic) - echo estimate
        E = D - Y

        # Overlap-save: take last N samples from the 2N IFFT output
        e_time = np.fft.irfft(E, n=2 * N)
        out_block = e_time[N:]

        # Update running power estimate
        self._P = 0.9 * self._P + 0.1 * np.abs(X) ** 2

        # NLMS weight update
        step = self._mu * E / (self._P + self._delta)
        for k in range(self._M):
            self._W[k] += step * np.conj(self._X_buf[k])

        return out_block
