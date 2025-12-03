FROM docker.io/rocm/dev-ubuntu-24.04:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt update && apt install -y \
    pkg-config \
    cmake \
    libfftw3-dev \
    hipfft-dev \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the fluidfft repository
COPY . /workspace/

# Create virtual environment
RUN python3 -m venv /opt/venv

# Set environment to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install Python build dependencies
RUN pip install \
    meson-python \
    numpy \
    cython \
    transonic \
    pytest

# Install fluidfft-builder
RUN pip install -e /workspace/plugins/fluidfft-builder

# Install base fluidfft package
RUN pip install /workspace

# Install fluidfft-hipfft plugin with HIP compiler
RUN cd /workspace/plugins/fluidfft-hipfft && \
    CC=hipcc \
    CXX=hipcc \
    CMAKE_PREFIX_PATH=/opt/rocm-7.1.1 \
    pip install .

# Set environment variables for runtime
ENV CC=hipcc
ENV CXX=hipcc
ENV CMAKE_PREFIX_PATH=/opt/rocm-7.1.1

# Verify installation
RUN python -c "import fluidfft_hipfft.fft2d.with_hipfft; print('Plugin installed successfully')"

# Default command
CMD ["/bin/bash"]
