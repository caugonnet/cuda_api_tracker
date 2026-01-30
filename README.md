# CUDA API Version Tracker

Tools to find when specific CUDA API calls were introduced or deprecated/removed by crawling the NVIDIA CUDA documentation archive.

## Tools

| Script | Purpose |
|--------|---------|
| `cuda_api_tracker.py` | Find when a specific API was introduced/removed |
| `cuda_api_changelog.py` | Generate a full changelog across a range of versions |

## Features

- 🔍 **Search for specific APIs**: Find when `cudaStreamGetDevice`, `cudaLaunchCooperativeKernelMultiDevice`, etc. were introduced or removed
- 📊 **Generate changelogs**: List all APIs added or removed across a range of CUDA versions
- 💾 **Caching**: Results are cached locally (`.cache/`) to speed up subsequent queries
- 🔄 **Supports both API types**: Runtime API (`cuda*`) and Driver API (`cu*`)
- ⚡ **Efficient search**: Starts from latest version and works backwards, stopping at boundaries

---

## cuda_api_tracker.py

Find when a specific API was introduced or removed.

### Usage

```bash
# Search for a Runtime API
./cuda_api_tracker.py cudaStreamGetDevice

# Search for a Driver API (auto-detected by 'cu' prefix)
./cuda_api_tracker.py cuStreamCreate

# Explicitly specify API type
./cuda_api_tracker.py cudaMalloc --api-type runtime
./cuda_api_tracker.py cuMemAlloc --api-type driver

# Verbose output showing each version checked
./cuda_api_tracker.py cudaStreamGetDevice -v

# Check all versions (not just until boundary)
./cuda_api_tracker.py cudaMalloc --full-scan
```

### Compare two versions

```bash
# List APIs added/removed between CUDA 11.0 and 12.0
./cuda_api_tracker.py --compare 11.0.3 12.0.0
```

### Example Output

```
$ ./cuda_api_tracker.py cudaLaunchCooperativeKernelMultiDevice

Searching for 'cudaLaunchCooperativeKernelMultiDevice' in runtime API documentation...
(Starting from latest version, working backwards)

  Checking CUDA 13.1.0 (latest)... ✗ NOT FOUND

  API not in latest. Checking if it was removed...
  Checking CUDA 11.8.0... ✓ FOUND

============================================================
API: cudaLaunchCooperativeKernelMultiDevice
Type: CUDA Runtime API
============================================================
(Checked 3 of 40 versions)

✅ INTRODUCED in: CUDA 9.0
❌ REMOVED/DEPRECATED after: CUDA 11.8.0
   (First missing in: CUDA 12.0.0)
```

---

## cuda_api_changelog.py

Generate a comprehensive changelog of all API additions and removals.

### Usage

```bash
# Generate changelog since CUDA 11.8
./cuda_api_changelog.py --since 11.8

# Generate changelog for a specific range
./cuda_api_changelog.py --since 11.0 --until 12.0

# Output as Markdown
./cuda_api_changelog.py --since 12.0 --format markdown --output changelog.md

# Output as JSON (for programmatic use)
./cuda_api_changelog.py --since 11.8 --format json --output changelog.json

# Output as CSV
./cuda_api_changelog.py --since 11.8 --format csv --output changes.csv

# Driver API changelog
./cuda_api_changelog.py --since 11.8 --api-type driver
```

### Output Formats

| Format | Description |
|--------|-------------|
| `text` | Human-readable plain text (default) |
| `markdown` | GitHub-flavored Markdown with collapsible sections |
| `json` | Structured JSON for programmatic use |
| `csv` | CSV format (version, action, api_name) |

### Example Output

```
$ ./cuda_api_changelog.py --since 12.0 --until 12.2

======================================================================
CUDA Runtime API Changelog
Versions: 12.0.0 → 12.2.2
======================================================================

SUMMARY
  Total APIs added:   15
  Total APIs removed: 3
  Net new APIs:       12
  Net removed APIs:   0

======================================================================
CHANGES BY VERSION
======================================================================

## CUDA 12.1.0 (from 12.0.1)
   Total APIs: 425

   ✅ ADDED (8):
      + cudaGraphAddMemcpyNode1D
      + cudaGraphAddMemsetNode
      ...

   ❌ REMOVED (2):
      - cudaOldDeprecatedFunc
      ...
```

---

## Cache Management

```bash
# Clear cached API data (forces re-download)
./cuda_api_tracker.py --clear-cache

# Run without using cache
./cuda_api_tracker.py cudaMalloc --no-cache
```

Cache is stored in `.cache/` in the script directory.

## Supported CUDA Versions

The scripts check CUDA versions from 7.5 through 13.1.0, including:
- Major releases
- Point releases

## Requirements

- Python 3.6+
- No external dependencies (uses only Python standard library)

## How It Works

1. Fetches CUDA Runtime/Driver API documentation from the NVIDIA documentation archive
2. Parses HTML to extract function names using HTML parsing and regex patterns
3. Tracks when APIs appear and disappear across versions
4. Caches results locally for faster subsequent queries

## Documentation Sources

- Latest documentation: https://docs.nvidia.com/cuda/
- Archive: https://developer.nvidia.com/cuda-toolkit-archive
- Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/
- Driver API: https://docs.nvidia.com/cuda/cuda-driver-api/

## License

MIT License - Feel free to use and modify as needed.
