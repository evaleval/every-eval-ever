#!/bin/bash
# Setup script for external schemas

set -e

echo "🔧 Setting up external schemas..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_SCHEMAS_DIR="$SCRIPT_DIR/external_schemas"

# Create external schemas directory
mkdir -p "$EXTERNAL_SCHEMAS_DIR"
cd "$EXTERNAL_SCHEMAS_DIR"

# Clone or update evalHub schema
if [ -d "evalHub" ]; then
    echo "📄 Updating existing evalHub schema..."
    cd evalHub
    git pull origin main
    cd ..
else
    echo "📥 Cloning evalHub schema..."
    git clone https://github.com/evaleval/evalHub.git
fi

echo "✅ Schema setup complete!"
echo ""
echo "Schema files available at:"
echo "  - $EXTERNAL_SCHEMAS_DIR/evalHub/schema/eval_types.py"
echo "  - $EXTERNAL_SCHEMAS_DIR/evalHub/schema/eval.schema.json"
echo ""
echo "Note: External schemas are not committed to this repository."
echo "Run this script again to update to the latest schema versions."
