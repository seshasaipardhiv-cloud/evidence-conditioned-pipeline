import zipfile
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_safe_path(base_path: Path, target_path: Path) -> bool:
    try:
        resolved_target = target_path.resolve()
        resolved_base = base_path.resolve()
        return resolved_target.is_relative_to(resolved_base)
    except Exception:
        return False

def safe_extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> Tuple[bool, str]:
    if not zip_path.exists():
        error_msg = f"ZIP file not found: {zip_path}"
        logger.error(error_msg)
        return False, error_msg

    extract_to.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file:
                error_msg = f"ZIP integrity check failed. Bad file: {bad_file}"
                logger.error(error_msg)
                return False, error_msg

            for member in zip_ref.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    error_msg = f"Path traversal attempt detected in ZIP member: {member}"
                    logger.error(error_msg)
                    return False, error_msg

                target_path = extract_to / member_path
                
                if not is_safe_path(extract_to, target_path):
                    error_msg = f"Unsafe extraction path detected for member: {member}"
                    logger.error(error_msg)
                    return False, error_msg

                if target_path.exists() and not overwrite:
                    logger.warning(f"File already exists, skipping: {target_path}")
                    continue

                if member.endswith('/'):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
        return True, "Extraction successful"
    except zipfile.BadZipFile as e:
        error_msg = f"Bad ZIP file {zip_path}: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error extracting {zip_path}: {e}"
        logger.error(error_msg)
        return False, error_msg
