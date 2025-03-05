#Credit: Debug using ChatGPT and Google
#!/bin/bash

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Add project root to PYTHONPATH so tests can find the src package
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run all tests in the tests directory
#python3 -m unittest discover -s "$PROJECT_ROOT/test" -v

# Run the Python unit tests
python3 -m unittest $PROJECT_ROOT/test/test_SimplifiedThreePL.py

# Check the exit status of the test run 
#if [ $? -eq 0 ]; then
#    echo "All tests passed!"
#else
#    echo "Some tests failed!"
#fi