#!/bin/bash

echo "=========================================="
echo "PIND-Quadruped Repository Verification"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found"
fi

# Check directory structure
echo ""
echo "Checking directory structure..."
DIRS=("pind" "pind/models" "pind/utils" "pind/data" "pind/configs" "scripts" "tests" "docs" "experiments" "data")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "  ${GREEN}✓${NC} $dir/"
    else
        echo -e "  ${RED}✗${NC} $dir/ missing"
    fi
done

# Check key files
echo ""
echo "Checking key files..."
FILES=("README.md" "setup.py" "requirements.txt" "LICENSE" "CITATION.bib" "run_experiments.py")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file missing"
    fi
done

# Check Python modules
echo ""
echo "Checking Python modules..."
MODULES=("pind/models/pind.py" "pind/models/baselines.py" "pind/utils/integrators.py" "pind/data/generator.py")
for module in "${MODULES[@]}"; do
    if [ -f "$module" ]; then
        echo -e "  ${GREEN}✓${NC} $module"
    else
        echo -e "  ${RED}✗${NC} $module missing"
    fi
done

# Check if package can be imported (if installed)
echo ""
echo "Checking package import..."
if python3 -c "import pind" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} pind package importable"
else
    echo -e "  ${YELLOW}!${NC} pind package not installed (run: pip install -e .)"
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install package: pip install -e ."
echo "  2. Generate data:   make data"
echo "  3. Train model:     make train"
echo "  4. Run benchmark:   make benchmark"
echo ""
echo "For detailed instructions, see:"
echo "  - README.md"
echo "  - docs/quickstart.md"
echo ""
