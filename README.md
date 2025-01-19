# CUDA Ethereum Vanity Address Generator

A high-performance, multi-GPU Ethereum vanity address generator implemented in CUDA. This tool can generate billions of Ethereum addresses per second to find addresses matching specific patterns or even crack specific target addresses.

## Features

- **High Performance**: Utilizes NVIDIA GPUs for parallel address generation and checking
- **Multi-GPU Support**: Scales linearly across multiple GPUs in the same system
- **Flexible Pattern Matching**: Search for addresses starting with specific patterns or match full addresses
- **Optimized Cryptography**: GPU-accelerated secp256k1 and Keccak-256 implementations
- **Real-time Statistics**: Monitor keys/second and total keys checked

## Requirements

- NVIDIA GPU with Compute Capability 7.5 or higher
- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- C++17 compatible compiler

## Building

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cuda-eth-cracker.git
   cd cuda-eth-cracker
   ```

2. Create build directory and build:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

The program accepts the following command line arguments:

```bash
./eth_cracker [options]
Options:
  -p, --pattern <pattern>    Target pattern to search for (e.g., '0x1234')
  -f, --full-address        Search for exact address match
  -h, --help               Show this help message
```

### Examples

1. Find an address starting with "1234":
   ```bash
   ./eth_cracker -p 1234
   ```

2. Find a specific address:
   ```bash
   ./eth_cracker -f -p 0x1234567890123456789012345678901234567890
   ```

## Performance Optimization

The tool is already optimized for high performance, but you can tune it further:

1. Adjust `BLOCK_SIZE` and `KEYS_PER_THREAD` in `gpu_manager.hpp` based on your GPU
2. Use `nvidia-smi` to monitor GPU utilization and memory usage
3. For multi-GPU systems, the workload is automatically balanced

## Implementation Details

The implementation uses several optimizations:

1. **Efficient Field Arithmetic**: Custom implementation of finite field operations for secp256k1
2. **Batch Processing**: Generates and checks multiple keys per thread
3. **Memory Coalescing**: Optimized memory access patterns for better throughput
4. **Warp-level Optimizations**: Uses warp-level primitives where possible

## Security Notes

1. Generated private keys are cryptographically secure using CUDA's random number generator
2. The tool can be used for both vanity address generation and address cracking
3. Use responsibly and be aware of the legal implications

## Contributing

Contributions are welcome! Please feel free to submit pull requests. Areas for improvement:

1. Additional cryptographic optimizations
2. Support for more pattern matching options
3. Integration with other blockchain networks
4. UI improvements and better statistics reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The secp256k1 implementation is based on the Bitcoin Core implementation
- Keccak-256 implementation follows the FIPS 202 standard
- CUDA optimization techniques from NVIDIA's best practices guide

## Disclaimer

This tool is for educational and research purposes only. The authors are not responsible for any misuse or damage caused by this program. 