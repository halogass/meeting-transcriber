import os
import tempfile

# keep tests away from the real data folder
os.environ.setdefault("MT_DATA_DIR", tempfile.mkdtemp(prefix="mt-test-"))
