import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_root = os.environ.get('B3_DATA_ROOT', os.path.join(PROJECT_ROOT_DIR, 'data'))

db_names = {'PASCALContext': 'PASCALContext', 'NYUD_MT': 'NYUDv2'}
db_paths = {}
for database, db_pa in db_names.items():
    db_paths[database] = os.path.join(db_root, db_pa)
