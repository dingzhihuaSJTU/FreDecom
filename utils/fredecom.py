import numpy as np
import torch

# ##########v4.1 均频率分割##########

# v4.1 频率均匀分割
def bandpass(freqs, fft, cutoff_frequency_low, cutoff_frequency_high):
    H = torch.ones_like(freqs, dtype=fft.dtype, device=fft.device)
    H[torch.abs(freqs) < cutoff_frequency_low] = 0
    H[torch.abs(freqs) >= cutoff_frequency_high] = 0
    return H.unsqueeze(0).unsqueeze(2) * fft

def SpectrumCentralization(fft, fshift):
    fft_left = torch.roll(fft, -int(fshift - 1))
    fft_right = torch.roll(fft, int(fshift - 1))
    fft_centered = torch.zeros_like(fft, dtype=fft.dtype, device=fft.device)
    N = fft.shape[1]
    fft_centered[..., :N // 2 - (fshift - 1), :] = fft_left[..., :N // 2 - (fshift - 1), :]
    fft_centered[..., N // 2 + (fshift - 1):, :] = fft_right[..., N // 2 + (fshift - 1):, :]
    return fft_centered

def InverseSpectrumCentralization(fft, fshift):
    if fshift == 1:
        return fft
    fft_left = torch.roll(fft, int(fshift - 1))
    fft_right = torch.roll(fft, int(-(fshift - 1)))
    fft_centered = torch.zeros_like(fft, dtype=fft.dtype, device=fft.device)
    N = fft.shape[1]
    fft_centered[..., (fshift - 1): N // 2, :] = fft_left[..., (fshift - 1): N // 2, :]
    fft_centered[..., N // 2: -(fshift - 1), :] = fft_right[..., N // 2: -(fshift - 1), :]
    return fft_centered

def FrequencyDecomposition(data: torch.Tensor, granularity: int = 10) -> torch.Tensor:
    """
    Perform frequency decomposition on input data with uniform energy segmentation.
    Supports batch processing with batch_size as the first dimension of input data.

    Args:
        data (torch.Tensor): Input data of shape (batch_size, dimofdata, lenofdata).
        granularity (int): Number of frequency bands to decompose into.

    Returns:
        torch.Tensor: Decomposed data of shape (batch_size, dimofdata, granularity, lenofdata).
    """

    # Ensure data has shape (batch_size, len, dim)
    if data.dim() == 2:  # Add batch_size dimension if missing
        data = data.unsqueeze(-1)

    batch_size, lenofdata, dimofdata = data.shape

    # Cut-off frequencies for uniform frequency segmentation
    k_c = torch.ceil(torch.tensor(lenofdata / 2 / granularity))

    freqs = torch.fft.fftfreq(lenofdata, d=1 / lenofdata, device=data.device)
    fft_data = torch.fft.fft(data, dim=1)  # FFT along the time dimension

    # Preallocate output tensor（batch_size, lenofdata, dimofdata, granularity）
    output = torch.zeros((batch_size, lenofdata, dimofdata, granularity), dtype=data.dtype, device=data.device)

    # Process each frequency band
    for i in range(granularity):
        minfre = int(i * k_c)
        maxfre = int(min((i + 1) * k_c, lenofdata // 2 + 1))

        # Apply bandpass filter
        fft_bandpass = bandpass(freqs, fft_data, minfre, maxfre)

        if i == 0:
            output[:, :, :, i] = torch.fft.ifft(fft_bandpass, dim=1)
        else:
            fft_centered = SpectrumCentralization(fft_bandpass, minfre)
            output[:, :, :, i] = torch.fft.ifft(fft_centered, dim=1)

    return output.real  # Return the real part of the decomposed data


def InverseFrequencyDecomposition(data: torch.Tensor, granularity: int) -> torch.Tensor:
    """
    Reconstruct the original time series from its frequency decomposition.

    Args:
        data (torch.Tensor): Decomposed data of shape (batch_size, dimofdata, granularity, lenofdata).

    Returns:
        torch.Tensor: Reconstructed data of shape (batch_size, dimofdata, lenofdata).
    """

    batch_size, lenofdata, dimofdata = data.shape
    dimofdata = int(dimofdata / granularity)
    data = data.reshape(batch_size, lenofdata, dimofdata, granularity)
    freqs = torch.fft.fftfreq(lenofdata, d=1 / lenofdata, device=data.device)

    # Cut-off frequencies for uniform frequency segmentation
    k_c = torch.ceil(torch.tensor(lenofdata / 2 / granularity))

    # Preallocate output tensor
    output = torch.zeros((batch_size, lenofdata, dimofdata), dtype=data.dtype, device=data.device)

    # Process each frequency band
    for i in range(granularity):
        if i == 0:
            output += data[:, :, :, i]
        else:
            minfre = int(i * k_c)
            maxfre = int(min((i + 1) * k_c, lenofdata // 2 + 1))
            fftdata = torch.fft.fft(data[:, :, :, i].to(torch.float32), dim=1)
            fftdata = bandpass(freqs, fftdata, 1, maxfre-minfre+1)
            fft_inver = InverseSpectrumCentralization(fftdata, minfre)
            output += torch.fft.ifft(fft_inver, dim=1).real

    return output