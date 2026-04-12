import json

# Load latest warmup log
with open('logs_warmup_woman_implicit_20260211_221337.json', 'r') as f:
    logs = json.load(f)

print('=== Models yang SUDAH ditest di warmup terakhir ===')
# Get unique tested model orders
tested_orders = set()
for entry in logs:
    tested_orders.add(entry['model_order'])
    status = 'OK' if entry.get('status') == 'SUCCESS' else 'FAIL'
    print(f"  {status}: order={entry['model_order']}, {entry['model_name']}")

# Remove duplicates - just show unique models
tested_orders_unique = sorted(tested_orders)
print(f'\nTested orders: {tested_orders_unique}')

print()
print('=== Models di config (order 31-49) ===')
with open('config/model_keys.json', 'r') as f:
    models = json.load(f)

new_models = []
for m in models:
    order = m.get('order', 0)
    if 31 <= order <= 49:
        tag = 'DONE' if order in tested_orders else 'NEW'
        print(f"  [{order}] {tag}: {m['model_id']}")
        if tag == 'NEW':
            new_models.append(m)

print()
print('=== Models BARU yang perlu ditest ===')
if new_models:
    for m in new_models:
        print(f"  [{m['order']}] {m['model_id']}")
else:
    print('  (tidak ada model baru)')
