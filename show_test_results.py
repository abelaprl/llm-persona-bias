import json

with open('logs_model_connectivity_test.json', 'r') as f:
    data = json.load(f)

success = [r for r in data if r['status'] == 'SUCCESS']
failed = [r for r in data if r['status'] == 'FAILED']

print("=" * 80)
print("MODEL CONNECTIVITY TEST RESULTS")
print("=" * 80)
print(f"\nTotal Models Tested: {len(data)}")
print(f"  ✓ Connected: {len(success)}")
print(f"  ✗ Failed: {len(failed)}")

print("\n" + "=" * 80)
print("SUCCESSFUL MODELS (Ready for experiments)")
print("=" * 80)
for r in success:
    print(f"\n[{r['order']:2d}] {r['name']}")
    print(f"     Model ID: {r['model_id']}")
    print(f"     Latency: {r['latency']:.2f}s")

print("\n" + "=" * 80)
print("FAILED MODELS (by reason)")
print("=" * 80)

# Group by error type
error_groups = {}
for r in failed:
    error_type = r['message'].split(':')[0]
    if error_type not in error_groups:
        error_groups[error_type] = []
    error_groups[error_type].append(r)

for error_type, models in sorted(error_groups.items()):
    print(f"\n{error_type}: {len(models)} models")
    for r in models[:3]:  # Show first 3
        print(f"  - [{r['order']:2d}] {r['name']}")
    if len(models) > 3:
        print(f"  ... and {len(models)-3} more")

print("\n" + "=" * 80)
