import numpy as np

def fourier_encode(x, num_frequencies=8):
    """
    Applies a Fourier transformation to the input x.
    It returns a representation using sine functions with exponential frequencies.
    This helps to encode continuous values into a richer representation.
    """
    frequencies = 2 ** np.arange(num_frequencies)  # Exponential frequency scaling: [1, 2, 4, 8, ...]
    encoded = np.sin(frequencies * x)  # Apply sine transformation
    return encoded