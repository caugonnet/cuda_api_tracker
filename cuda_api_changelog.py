#!/usr/bin/env python3
"""
CUDA API Catalog & Changelog Generator

Default: List ALL APIs present in a version range with their introduction/removal dates.
With --changelog: Show what changed between each consecutive version.

Usage:
    ./cuda_api_changelog.py --since 11.8              # List all APIs with lifecycle info
    ./cuda_api_changelog.py --since 11.8 --changelog  # Show version-by-version changes
    
Examples:
    ./cuda_api_changelog.py --since 11.8.0
    ./cuda_api_changelog.py --since 11.0.3 --until 12.0.0 --api-type driver
    ./cuda_api_changelog.py --since 12.0.0 --changelog --format markdown
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Import from the main tracker script
from cuda_api_tracker import (
    CUDA_VERSIONS,
    get_api_list_for_version,
)


def find_closest_version(target: str) -> str:
    """Find the closest matching version in CUDA_VERSIONS."""
    if target in CUDA_VERSIONS:
        return target
    
    for v in CUDA_VERSIONS:
        if v.startswith(target):
            return v
    
    target_parts = [int(x) for x in target.split('.')]
    for v in CUDA_VERSIONS:
        v_parts = [int(x) for x in v.split('.')]
        if v_parts[0] == target_parts[0]:
            if len(target_parts) == 1 or v_parts[1] >= target_parts[1]:
                return v
    
    return None


def get_version_range(since: str = None, until: str = None) -> list:
    """Get list of versions in the specified range."""
    if since is None:
        start_idx = 0
    else:
        since_v = find_closest_version(since)
        if not since_v:
            print(f"Error: Could not find version matching '{since}'")
            print(f"Available versions: {', '.join(CUDA_VERSIONS)}")
            sys.exit(1)
        start_idx = CUDA_VERSIONS.index(since_v)
    
    if until:
        until_v = find_closest_version(until)
        if not until_v:
            print(f"Error: Could not find version matching '{until}'")
            sys.exit(1)
        end_idx = CUDA_VERSIONS.index(until_v) + 1
    else:
        end_idx = len(CUDA_VERSIONS)
    
    return CUDA_VERSIONS[start_idx:end_idx]


def fetch_all_versions(versions: list, api_types: list) -> dict:
    """Fetch API lists for all versions. Returns {version: set(apis)}."""
    version_apis = {}
    for i, version in enumerate(versions):
        print(f"  [{i+1}/{len(versions)}] Fetching CUDA {version}...", end=" ", flush=True)
        apis = set()
        for api_type in api_types:
            apis.update(get_api_list_for_version(version, api_type))
        version_apis[version] = apis
        print(f"({len(apis)} APIs)")
    return version_apis


def generate_api_catalog(since: str, until: str = None, api_types: list = None) -> dict:
    """
    Generate a catalog of ALL APIs with their lifecycle info.
    
    Returns:
        Dictionary with all APIs and when they were introduced/removed
    """
    if api_types is None:
        api_types = ["runtime", "driver"]
    
    versions = get_version_range(since, until)
    
    api_type_str = " + ".join(api_types) if len(api_types) > 1 else api_types[0]
    print(f"Generating {api_type_str} API catalog for CUDA {versions[0]} → {versions[-1]}")
    print(f"Checking {len(versions)} versions...\n")
    
    version_apis = fetch_all_versions(versions, api_types)
    
    # Build catalog: track when each API was introduced and removed
    all_apis = set()
    for apis in version_apis.values():
        all_apis.update(apis)
    
    catalog = {
        'api_types': api_types,
        'api_type': api_type_str,  # for display
        'since': versions[0],
        'until': versions[-1],
        'generated': datetime.now().isoformat(),
        'total_apis': len(all_apis),
        'apis': {}
    }
    
    for api in sorted(all_apis):
        # Find first version where API appears
        first_seen = None
        removed = None
        present_in = []
        
        for version in versions:
            if api in version_apis[version]:
                present_in.append(version)
                if first_seen is None:
                    first_seen = version
            else:
                # If we had it before but not now, it was removed
                if present_in and removed is None:
                    removed = version
        
        # Determine status
        if api in version_apis[versions[-1]]:
            status = "present"
        elif present_in:
            status = "removed"
        else:
            status = "never_present"  # shouldn't happen
        
        # Only show "introduced" if it was added AFTER the first version in range
        # If it was already present in the first version, we don't know when it was actually introduced
        introduced = first_seen if first_seen != versions[0] else None
        
        catalog['apis'][api] = {
            'introduced': introduced,
            'removed': removed,
            'status': status,
            'present_in': present_in
        }
    
    # Summary stats
    catalog['summary'] = {
        'total': len(all_apis),
        'present': sum(1 for a in catalog['apis'].values() if a['status'] == 'present'),
        'removed': sum(1 for a in catalog['apis'].values() if a['status'] == 'removed'),
        'introduced_in_range': sum(1 for a in catalog['apis'].values() 
                                   if a['introduced'] is not None),
        'already_present': sum(1 for a in catalog['apis'].values() 
                               if a['introduced'] is None and a['present_in']),
    }
    
    return catalog


def generate_changelog(since: str, until: str = None, api_types: list = None, 
                       verbose: bool = False) -> dict:
    """
    Generate a changelog of API additions and removals (version-by-version diff).
    """
    if api_types is None:
        api_types = ["runtime", "driver"]
    
    versions = get_version_range(since, until)
    
    if len(versions) < 2:
        print("Error: Need at least 2 versions to generate changelog")
        sys.exit(1)
    
    api_type_str = " + ".join(api_types) if len(api_types) > 1 else api_types[0]
    print(f"Generating {api_type_str} API changelog for CUDA {versions[0]} → {versions[-1]}")
    print(f"Checking {len(versions)} versions...\n")
    
    changelog = {
        'api_types': api_types,
        'api_type': api_type_str,
        'since': versions[0],
        'until': versions[-1],
        'generated': datetime.now().isoformat(),
        'versions': [],
        'summary': {
            'total_added': 0,
            'total_removed': 0,
            'all_added': [],
            'all_removed': []
        }
    }
    
    prev_apis = None
    prev_version = None
    
    for i, version in enumerate(versions):
        print(f"  [{i+1}/{len(versions)}] Fetching CUDA {version}...", end=" ", flush=True)
        
        apis = set()
        for api_type in api_types:
            apis.update(get_api_list_for_version(version, api_type))
        print(f"({len(apis)} APIs)")
        
        if prev_apis is not None:
            added = sorted(apis - prev_apis)
            removed = sorted(prev_apis - apis)
            
            version_info = {
                'version': version,
                'previous': prev_version,
                'total_apis': len(apis),
                'added': added,
                'removed': removed,
                'added_count': len(added),
                'removed_count': len(removed)
            }
            
            changelog['versions'].append(version_info)
            changelog['summary']['total_added'] += len(added)
            changelog['summary']['total_removed'] += len(removed)
            changelog['summary']['all_added'].extend(added)
            changelog['summary']['all_removed'].extend(removed)
            
            if verbose and (added or removed):
                if added:
                    print(f"      + {len(added)} added: {', '.join(added[:5])}{'...' if len(added) > 5 else ''}")
                if removed:
                    print(f"      - {len(removed)} removed: {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
        
        prev_apis = apis
        prev_version = version
    
    changelog['summary']['all_added'] = sorted(set(changelog['summary']['all_added']))
    changelog['summary']['all_removed'] = sorted(set(changelog['summary']['all_removed']))
    
    net_new = set(changelog['summary']['all_added']) - set(changelog['summary']['all_removed'])
    net_removed = set(changelog['summary']['all_removed']) - set(changelog['summary']['all_added'])
    changelog['summary']['net_new'] = sorted(net_new)
    changelog['summary']['net_removed'] = sorted(net_removed)
    
    return changelog


# ============================================================================
# Catalog formatters (default mode - list all APIs)
# ============================================================================

def format_catalog_text(catalog: dict) -> str:
    """Format API catalog as plain text."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"CUDA {catalog['api_type'].title()} API Catalog")
    lines.append(f"Versions: {catalog['since']} → {catalog['until']}")
    lines.append(f"Generated: {catalog['generated']}")
    lines.append("=" * 80)
    
    lines.append(f"\nSUMMARY")
    lines.append(f"  Total APIs found:       {catalog['summary']['total']}")
    lines.append(f"  Already present in {catalog['since']}:  {catalog['summary']['already_present']}")
    lines.append(f"  Introduced in range:    {catalog['summary']['introduced_in_range']}")
    lines.append(f"  Removed in range:       {catalog['summary']['removed']}")
    lines.append(f"  Currently present:      {catalog['summary']['present']}")
    
    lines.append(f"\n{'=' * 80}")
    lines.append(f"{'API Name':<50} {'Introduced':<12} {'Removed':<12} {'Status'}")
    lines.append("=" * 80)
    
    for api, info in catalog['apis'].items():
        introduced = info['introduced'] or '-'
        removed = info['removed'] or '-'
        status = '✓' if info['status'] == 'present' else '✗ removed'
        lines.append(f"{api:<50} {introduced:<12} {removed:<12} {status}")
    
    return '\n'.join(lines)


def format_catalog_markdown(catalog: dict) -> str:
    """Format API catalog as Markdown."""
    lines = []
    lines.append(f"# CUDA {catalog['api_type'].title()} API Catalog")
    lines.append(f"\n**Versions:** {catalog['since']} → {catalog['until']}  ")
    lines.append(f"**Generated:** {catalog['generated']}")
    
    lines.append(f"\n## Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total APIs found | {catalog['summary']['total']} |")
    lines.append(f"| Already present in {catalog['since']} | {catalog['summary']['already_present']} |")
    lines.append(f"| Introduced in range | {catalog['summary']['introduced_in_range']} |")
    lines.append(f"| Removed in range | {catalog['summary']['removed']} |")
    lines.append(f"| Currently present | {catalog['summary']['present']} |")
    
    # Group by status
    present_apis = {k: v for k, v in catalog['apis'].items() if v['status'] == 'present'}
    removed_apis = {k: v for k, v in catalog['apis'].items() if v['status'] == 'removed'}
    
    lines.append(f"\n## All APIs ({len(catalog['apis'])})\n")
    lines.append(f"| API | Introduced | Removed | Status |")
    lines.append(f"|-----|------------|---------|--------|")
    
    for api, info in catalog['apis'].items():
        introduced = info['introduced'] or '-'
        removed = info['removed'] or '-'
        status = '✅' if info['status'] == 'present' else '❌'
        lines.append(f"| `{api}` | {introduced} | {removed} | {status} |")
    
    return '\n'.join(lines)


def format_catalog_csv(catalog: dict) -> str:
    """Format API catalog as CSV."""
    lines = []
    lines.append("api_name,introduced,removed,status,present_in_versions")
    
    for api, info in catalog['apis'].items():
        introduced = info['introduced'] or ''
        removed = info['removed'] or ''
        status = info['status']
        present_in = ';'.join(info['present_in'])
        lines.append(f"{api},{introduced},{removed},{status},{present_in}")
    
    return '\n'.join(lines)


# ============================================================================
# Changelog formatters (--changelog mode)
# ============================================================================

def format_changelog_text(changelog: dict) -> str:
    """Format changelog as plain text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"CUDA {changelog['api_type'].title()} API Changelog")
    lines.append(f"Versions: {changelog['since']} → {changelog['until']}")
    lines.append(f"Generated: {changelog['generated']}")
    lines.append("=" * 70)
    
    lines.append(f"\nSUMMARY")
    lines.append(f"  Total APIs added:   {changelog['summary']['total_added']}")
    lines.append(f"  Total APIs removed: {changelog['summary']['total_removed']}")
    lines.append(f"  Net new APIs:       {len(changelog['summary']['net_new'])}")
    lines.append(f"  Net removed APIs:   {len(changelog['summary']['net_removed'])}")
    
    lines.append(f"\n{'=' * 70}")
    lines.append("CHANGES BY VERSION")
    lines.append("=" * 70)
    
    for v in changelog['versions']:
        if v['added'] or v['removed']:
            lines.append(f"\n## CUDA {v['version']} (from {v['previous']})")
            lines.append(f"   Total APIs: {v['total_apis']}")
            
            if v['added']:
                lines.append(f"\n   ✅ ADDED ({v['added_count']}):")
                for api in v['added']:
                    lines.append(f"      + {api}")
            
            if v['removed']:
                lines.append(f"\n   ❌ REMOVED ({v['removed_count']}):")
                for api in v['removed']:
                    lines.append(f"      - {api}")
    
    if changelog['summary']['net_new']:
        lines.append(f"\n{'=' * 70}")
        lines.append(f"ALL NET NEW APIs ({len(changelog['summary']['net_new'])})")
        lines.append("=" * 70)
        for api in changelog['summary']['net_new']:
            lines.append(f"  + {api}")
    
    if changelog['summary']['net_removed']:
        lines.append(f"\n{'=' * 70}")
        lines.append(f"ALL NET REMOVED APIs ({len(changelog['summary']['net_removed'])})")
        lines.append("=" * 70)
        for api in changelog['summary']['net_removed']:
            lines.append(f"  - {api}")
    
    return '\n'.join(lines)


def format_changelog_markdown(changelog: dict) -> str:
    """Format changelog as Markdown."""
    lines = []
    lines.append(f"# CUDA {changelog['api_type'].title()} API Changelog")
    lines.append(f"\n**Versions:** {changelog['since']} → {changelog['until']}  ")
    lines.append(f"**Generated:** {changelog['generated']}")
    
    lines.append(f"\n## Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total APIs added | {changelog['summary']['total_added']} |")
    lines.append(f"| Total APIs removed | {changelog['summary']['total_removed']} |")
    lines.append(f"| Net new APIs | {len(changelog['summary']['net_new'])} |")
    lines.append(f"| Net removed APIs | {len(changelog['summary']['net_removed'])} |")
    
    lines.append(f"\n## Changes by Version\n")
    
    for v in changelog['versions']:
        if v['added'] or v['removed']:
            lines.append(f"\n### CUDA {v['version']}")
            lines.append(f"\n*From {v['previous']} • {v['total_apis']} total APIs*\n")
            
            if v['added']:
                lines.append(f"\n<details>")
                lines.append(f"<summary>✅ Added ({v['added_count']})</summary>\n")
                lines.append("```")
                for api in v['added']:
                    lines.append(api)
                lines.append("```")
                lines.append("</details>")
            
            if v['removed']:
                lines.append(f"\n<details>")
                lines.append(f"<summary>❌ Removed ({v['removed_count']})</summary>\n")
                lines.append("```")
                for api in v['removed']:
                    lines.append(api)
                lines.append("```")
                lines.append("</details>")
    
    if changelog['summary']['net_new']:
        lines.append(f"\n## All Net New APIs ({len(changelog['summary']['net_new'])})\n")
        lines.append("<details>")
        lines.append("<summary>Click to expand</summary>\n")
        lines.append("```")
        for api in changelog['summary']['net_new']:
            lines.append(api)
        lines.append("```")
        lines.append("</details>")
    
    if changelog['summary']['net_removed']:
        lines.append(f"\n## All Net Removed APIs ({len(changelog['summary']['net_removed'])})\n")
        lines.append("<details>")
        lines.append("<summary>Click to expand</summary>\n")
        lines.append("```")
        for api in changelog['summary']['net_removed']:
            lines.append(api)
        lines.append("```")
        lines.append("</details>")
    
    return '\n'.join(lines)


def format_changelog_csv(changelog: dict) -> str:
    """Format changelog as CSV."""
    lines = []
    lines.append("version,previous_version,action,api_name")
    
    for v in changelog['versions']:
        for api in v['added']:
            lines.append(f"{v['version']},{v['previous']},added,{api}")
        for api in v['removed']:
            lines.append(f"{v['version']},{v['previous']},removed,{api}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA API catalog or changelog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Default:     List ALL APIs with when they were introduced/removed
  --changelog: Show what changed between each version (diff view)

Examples:
  %(prog)s                                       # All APIs from all versions
  %(prog)s --since 11.8                          # All APIs since CUDA 11.8
  %(prog)s --since 11.8 --changelog              # Version-by-version changes
  %(prog)s --since 11.0 --until 12.0 --format csv -o apis.csv
  %(prog)s --changelog --format markdown         # Full changelog as markdown
        """
    )
    
    parser.add_argument('--since', default=None,
                       help='Starting CUDA version (default: oldest available)')
    parser.add_argument('--until',
                       help='Ending CUDA version (default: latest)')
    parser.add_argument('--api-type', choices=['runtime', 'driver', 'both'], default='both',
                       help='API type: runtime, driver, or both (default: both)')
    parser.add_argument('--changelog', action='store_true',
                       help='Show version-by-version changes instead of full API list')
    parser.add_argument('--format', '-f', choices=['text', 'markdown', 'json', 'csv'],
                       default='text', help='Output format (default: text)')
    parser.add_argument('--output', '-o',
                       help='Output file (default: stdout)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed progress')
    
    args = parser.parse_args()
    
    # Convert api_type to list
    if args.api_type == 'both':
        api_types = ['runtime', 'driver']
    else:
        api_types = [args.api_type]
    
    if args.changelog:
        # Changelog mode: version-by-version diff
        data = generate_changelog(args.since, args.until, api_types, verbose=args.verbose)
        
        if args.format == 'json':
            output = json.dumps(data, indent=2)
        elif args.format == 'markdown':
            output = format_changelog_markdown(data)
        elif args.format == 'csv':
            output = format_changelog_csv(data)
        else:
            output = format_changelog_text(data)
    else:
        # Default mode: full API catalog with lifecycle info
        data = generate_api_catalog(args.since, args.until, api_types)
        
        if args.format == 'json':
            output = json.dumps(data, indent=2)
        elif args.format == 'markdown':
            output = format_catalog_markdown(data)
        elif args.format == 'csv':
            output = format_catalog_csv(data)
        else:
            output = format_catalog_text(data)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"\n✅ Output written to {args.output}")
    else:
        print("\n")
        print(output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
