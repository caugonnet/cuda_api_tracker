#!/usr/bin/env python3
"""
CUDA API Version Tracker

This script crawls the NVIDIA CUDA documentation archive to find when
specific API calls were introduced or deprecated/removed.

Usage:
    python cuda_api_tracker.py <api_name> [--api-type runtime|driver]
    
Examples:
    python cuda_api_tracker.py cudaStreamGetDevice
    python cuda_api_tracker.py cudaLaunchCooperativeKernelMultiDevice
    python cuda_api_tracker.py cuStreamGetDevice --api-type driver
"""

import argparse
import re
import sys
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from html.parser import HTMLParser
import time

# CUDA versions to check (from oldest to newest)
# Based on https://developer.nvidia.com/cuda-toolkit-archive
# Only includes versions with online documentation available
CUDA_VERSIONS = [
    # CUDA 8.x-10.x use short version format (major.minor)
    "8.0",
    "9.0", "9.1", "9.2",
    "10.0", "10.1", "10.2",
    # CUDA 11.0 uses short format, 11.1+ uses full version
    "11.0",
    "11.1.0", "11.1.1",
    "11.2.0", "11.2.1", "11.2.2",
    "11.3.0", "11.3.1",
    "11.4.0", "11.4.1", "11.4.2", "11.4.3", "11.4.4",
    "11.5.0", "11.5.1", "11.5.2",
    "11.6.0", "11.6.1", "11.6.2",
    "11.7.0", "11.7.1",
    "11.8.0",
    # CUDA 12.x+
    "12.0.0", "12.0.1",
    "12.1.0", "12.1.1",
    "12.2.0", "12.2.1", "12.2.2",
    "12.3.0", "12.3.1", "12.3.2",
    "12.4.0", "12.4.1",
    "12.5.0", "12.5.1",
    "12.6.0", "12.6.1", "12.6.2", "12.6.3",
    "12.8.0", "12.8.1",
    "12.9.0", "12.9.1",
    "13.0.0", "13.0.1", "13.0.2",
    "13.1.0", "13.1.1",
    # Note: The last version may use main docs URL if not yet in archive
]

# Base URLs for documentation
ARCHIVE_BASE = "https://docs.nvidia.com/cuda/archive"
LATEST_BASE = "https://docs.nvidia.com/cuda"

# Cache directory for downloaded API lists (local to script directory)
CACHE_DIR = Path(__file__).parent / ".cache"


class CUDAAPIParser(HTMLParser):
    """Parse CUDA documentation HTML to extract API function names."""
    
    def __init__(self):
        super().__init__()
        self.api_names = set()
        self.in_link = False
        self.current_href = ""
        
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            # Look for links to API function documentation
            # These typically contain patterns like "group__CUDART" or "group__CUDA"
            if "group__" in href or "#" in href:
                self.in_link = True
                self.current_href = href
                
    def handle_endtag(self, tag):
        if tag == "a":
            self.in_link = False
            self.current_href = ""
            
    def handle_data(self, data):
        if self.in_link:
            data = data.strip()
            # Match CUDA API function naming patterns
            # Runtime API: cuda* (e.g., cudaMalloc, cudaStreamCreate)
            # Driver API: cu* (e.g., cuMemAlloc, cuStreamCreate)
            if re.match(r'^(cuda|cu)[A-Z][a-zA-Z0-9_]*$', data):
                self.api_names.add(data)


class ModuleIndexParser(HTMLParser):
    """Parse the modules index page to find all API group pages."""
    
    def __init__(self):
        super().__init__()
        self.group_links = []
        
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if "group__" in href and href not in self.group_links:
                self.group_links.append(href)


def get_cache_path(version: str, api_type: str) -> Path:
    """Get the cache file path for a specific version and API type."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{api_type}_{version.replace('.', '_')}.json"


def fetch_url(url: str, timeout: int = 30, retries: int = 3) -> str:
    """Fetch URL content with error handling and retries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }
    request = Request(url, headers=headers)
    
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode('utf-8', errors='ignore')
        except (HTTPError, URLError) as e:
            return ""
        except (IncompleteRead, ConnectionResetError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))  # exponential backoff
                continue
            return ""
        except Exception as e:
            # Catch any other network-related errors
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return ""
    
    return ""


def extract_apis_from_html(html: str) -> set:
    """Extract API names from HTML content."""
    parser = CUDAAPIParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    
    # Also use regex as fallback to catch more API names
    # Look for function definitions and references
    patterns = [
        r'\b(cuda[A-Z][a-zA-Z0-9_]*)\s*\(',  # cudaFunctionName(
        r'\b(cu[A-Z][a-zA-Z0-9_]*)\s*\(',     # cuFunctionName(
        r'>(cuda[A-Z][a-zA-Z0-9_]*)<',         # >cudaFunctionName<
        r'>(cu[A-Z][a-zA-Z0-9_]*)<',           # >cuFunctionName<
        r'"(cuda[A-Z][a-zA-Z0-9_]*)"',         # "cudaFunctionName"
        r'"(cu[A-Z][a-zA-Z0-9_]*)"',           # "cuFunctionName"
    ]
    
    apis = set(parser.api_names)
    for pattern in patterns:
        matches = re.findall(pattern, html)
        apis.update(matches)
    
    return apis


def get_api_list_for_version(version: str, api_type: str = "runtime", use_cache: bool = True) -> set:
    """
    Get the list of API functions for a specific CUDA version.
    
    Args:
        version: CUDA version string (e.g., "12.0.0")
        api_type: "runtime" or "driver"
        use_cache: Whether to use cached results
        
    Returns:
        Set of API function names
    """
    cache_path = get_cache_path(version, api_type)
    
    # Check cache first
    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                return set(json.load(f))
        except Exception:
            pass
    
    # Determine the documentation URL
    if api_type == "runtime":
        api_doc = "cuda-runtime-api"
    else:
        api_doc = "cuda-driver-api"
    
    # Try archive URL first
    urls_to_try = [
        f"{ARCHIVE_BASE}/{version}/{api_doc}/index.html",
        f"{ARCHIVE_BASE}/{version}/{api_doc}/cuda-runtime-api/index.html",
    ]
    
    # For the latest version in our list, also try the main docs URL 
    # (in case it's not yet in the archive)
    if version == CUDA_VERSIONS[-1]:
        urls_to_try.append(f"{LATEST_BASE}/{api_doc}/index.html")
    
    apis = set()
    
    for base_url in urls_to_try:
        html = fetch_url(base_url)
        if not html:
            continue
            
        # Extract APIs from main page
        apis.update(extract_apis_from_html(html))
        
        # Find and fetch group pages (API categories)
        parser = ModuleIndexParser()
        try:
            parser.feed(html)
        except Exception:
            pass
        
        # Fetch each group page for more complete coverage
        base_path = base_url.rsplit('/', 1)[0]
        for link in parser.group_links[:20]:  # Limit to avoid too many requests
            if link.startswith('http'):
                group_url = link
            else:
                group_url = f"{base_path}/{link}"
            
            group_html = fetch_url(group_url)
            if group_html:
                apis.update(extract_apis_from_html(group_html))
        
        if apis:
            break
    
    # Also try the deprecated functions page
    deprecated_url = f"{ARCHIVE_BASE}/{version}/{api_doc}/group__CUDART__HIGHLEVEL.html"
    deprecated_html = fetch_url(deprecated_url)
    if deprecated_html:
        apis.update(extract_apis_from_html(deprecated_html))
    
    # Cache the results
    if apis and use_cache:
        try:
            with open(cache_path, 'w') as f:
                json.dump(sorted(apis), f)  # sorted for consistent git diffs
        except Exception:
            pass
    
    return apis


def find_api_history(api_name: str, api_type: str = "runtime", verbose: bool = False, full_scan: bool = False) -> dict:
    """
    Find when an API was introduced and/or removed.
    
    Strategy: Start from latest version and work backwards.
    - If present in latest: find when it was introduced (first version where it appeared)
    - If NOT present in latest: find when it was removed (last version where it existed)
    
    Args:
        api_name: Name of the CUDA API function
        api_type: "runtime" or "driver"
        verbose: Print progress information
        full_scan: Check all versions instead of stopping at boundary
        
    Returns:
        Dictionary with 'introduced', 'removed', and 'present_in' keys
    """
    result = {
        'api_name': api_name,
        'api_type': api_type,
        'introduced': None,
        'removed': None,
        'present_in': [],
        'not_found_in': [],
        'versions_checked': 0
    }
    
    print(f"Searching for '{api_name}' in {api_type} API documentation...")
    print(f"(Starting from latest version, working backwards)\n")
    
    # Work backwards from latest version
    versions_reversed = list(reversed(CUDA_VERSIONS))
    
    # First, check the latest version to determine our search strategy
    latest = versions_reversed[0]
    print(f"  Checking CUDA {latest} (latest)...", end=" ", flush=True)
    
    latest_apis = get_api_list_for_version(latest, api_type)
    result['versions_checked'] += 1
    
    api_in_latest = api_name in latest_apis
    
    if api_in_latest:
        print("✓ FOUND")
        result['present_in'].append(latest)
        
        # API exists in latest - find when it was introduced
        print(f"\n  API present in latest. Finding when it was introduced...")
        
        for version in versions_reversed[1:]:
            if verbose:
                print(f"  Checking CUDA {version}...", end=" ", flush=True)
            
            apis = get_api_list_for_version(version, api_type)
            result['versions_checked'] += 1
            
            if api_name in apis:
                result['present_in'].append(version)
                if verbose:
                    print("✓ FOUND")
                if not full_scan:
                    # Continue to find the actual introduction point
                    pass
            else:
                result['not_found_in'].append(version)
                if verbose:
                    print("✗ not found")
                if not full_scan:
                    # Found the boundary - API was introduced in the next version
                    break
        
        # Sort present_in from oldest to newest
        result['present_in'] = sorted(result['present_in'], 
                                       key=lambda v: CUDA_VERSIONS.index(v))
        result['introduced'] = result['present_in'][0]
        
    else:
        print("✗ NOT FOUND")
        result['not_found_in'].append(latest)
        
        # API not in latest - either removed or never existed
        print(f"\n  API not in latest. Checking if it was removed...")
        
        for version in versions_reversed[1:]:
            if verbose:
                print(f"  Checking CUDA {version}...", end=" ", flush=True)
            
            apis = get_api_list_for_version(version, api_type)
            result['versions_checked'] += 1
            
            if api_name in apis:
                result['present_in'].append(version)
                if verbose:
                    print("✓ FOUND")
                
                if not full_scan:
                    # Found when it existed - now find when it was introduced
                    print(f"\n  Found in CUDA {version}. Finding when it was introduced...")
                    
                    for older_version in versions_reversed[versions_reversed.index(version)+1:]:
                        if verbose:
                            print(f"  Checking CUDA {older_version}...", end=" ", flush=True)
                        
                        older_apis = get_api_list_for_version(older_version, api_type)
                        result['versions_checked'] += 1
                        
                        if api_name in older_apis:
                            result['present_in'].append(older_version)
                            if verbose:
                                print("✓ FOUND")
                        else:
                            result['not_found_in'].append(older_version)
                            if verbose:
                                print("✗ not found")
                            break
                    break
            else:
                result['not_found_in'].append(version)
                if verbose:
                    print("✗ not found")
        
        # Sort present_in from oldest to newest
        result['present_in'] = sorted(result['present_in'], 
                                       key=lambda v: CUDA_VERSIONS.index(v))
        
        if result['present_in']:
            result['introduced'] = result['present_in'][0]
            # Find the first version after the last present where it's missing
            last_present_idx = CUDA_VERSIONS.index(result['present_in'][-1])
            if last_present_idx + 1 < len(CUDA_VERSIONS):
                result['removed'] = CUDA_VERSIONS[last_present_idx + 1]
    
    return result


def print_result(result: dict):
    """Pretty print the search result."""
    print("\n" + "=" * 60)
    print(f"API: {result['api_name']}")
    print(f"Type: CUDA {result['api_type'].title()} API")
    print("=" * 60)
    
    versions_checked = result.get('versions_checked', len(CUDA_VERSIONS))
    print(f"(Checked {versions_checked} of {len(CUDA_VERSIONS)} versions)")
    
    if not result['present_in']:
        print("\n⚠️  API NOT FOUND in any checked version!")
        print("\nPossible reasons:")
        print("  - The API name might be misspelled")
        print("  - It might be in the other API type (runtime vs driver)")
        print("  - It might be in a version not checked (try --full-scan)")
        print("  - The documentation structure might have changed")
        return
    
    print(f"\n✅ INTRODUCED in: CUDA {result['introduced']}")
    
    if result['removed']:
        print(f"❌ REMOVED/DEPRECATED after: CUDA {result['present_in'][-1]}")
        print(f"   (First missing in: CUDA {result['removed']})")
    else:
        print(f"📌 STILL PRESENT in: CUDA {result['present_in'][-1]}")
    
    if len(result['present_in']) > 1:
        print(f"\nConfirmed in {len(result['present_in'])} versions:")
        # Group consecutive versions for cleaner output
        if len(result['present_in']) > 5:
            print(f"  {result['present_in'][0]} → {result['present_in'][-1]}")
        else:
            print(f"  {', '.join(result['present_in'])}")


def list_new_apis(version1: str, version2: str, api_type: str = "runtime"):
    """List APIs that were added between two versions."""
    print(f"Comparing CUDA {version1} → {version2} ({api_type} API)...")
    
    apis_v1 = get_api_list_for_version(version1, api_type)
    apis_v2 = get_api_list_for_version(version2, api_type)
    
    new_apis = apis_v2 - apis_v1
    removed_apis = apis_v1 - apis_v2
    
    print(f"\n📥 NEW APIs in CUDA {version2} ({len(new_apis)}):")
    for api in sorted(new_apis):
        print(f"  + {api}")
    
    print(f"\n📤 REMOVED APIs after CUDA {version1} ({len(removed_apis)}):")
    for api in sorted(removed_apis):
        print(f"  - {api}")


def clear_cache():
    """Clear the cached API lists."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}")
    else:
        print("No cache to clear.")


def main():
    parser = argparse.ArgumentParser(
        description="Track CUDA API version history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s cudaStreamGetDevice
  %(prog)s cudaLaunchCooperativeKernelMultiDevice
  %(prog)s cuStreamCreate --api-type driver
  %(prog)s --compare 11.0.3 12.0.0
  %(prog)s cudaMalloc --full-scan    # Check all versions
  %(prog)s --clear-cache
        """
    )
    
    parser.add_argument('api_name', nargs='?', help='Name of the CUDA API function to search for')
    parser.add_argument('--api-type', choices=['runtime', 'driver'], default='runtime',
                       help='API type to search (default: runtime)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed progress')
    parser.add_argument('--compare', nargs=2, metavar=('V1', 'V2'),
                       help='Compare two versions and list API changes')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cached API data')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--full-scan', action='store_true',
                       help='Check all versions instead of stopping at boundary')
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_cache()
        return 0
    
    if args.compare:
        list_new_apis(args.compare[0], args.compare[1], args.api_type)
        return 0
    
    if not args.api_name:
        parser.print_help()
        return 1
    
    # Auto-detect API type based on naming convention
    api_type = args.api_type
    if args.api_name.startswith('cu') and not args.api_name.startswith('cuda'):
        api_type = 'driver'
        print(f"(Auto-detected driver API based on 'cu' prefix)")
    
    result = find_api_history(args.api_name, api_type, verbose=args.verbose, full_scan=args.full_scan)
    print_result(result)
    
    # If not found, suggest trying the other API type
    if not result['present_in'] and api_type == 'runtime':
        print(f"\n💡 Tip: Try searching in the driver API:")
        print(f"   python {sys.argv[0]} {args.api_name} --api-type driver")
    
    return 0 if result['present_in'] else 1


if __name__ == "__main__":
    sys.exit(main())
