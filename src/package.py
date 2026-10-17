"""Package only this research project; never include raw data or the Python env."""
import json
import zipfile
from pathlib import Path

from .data import ROOT,sha256,dump


def main():
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink(): continue
        if any(x in rel.parts for x in ['.git','.venv','data','cache','__pycache__','.pytest_cache']): continue
        if path.suffix in ['.zip','.pyc'] or path.name=='PACKAGE_MANIFEST.json': continue
        files.append(path)
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in files}
    dump(ROOT/'PACKAGE_MANIFEST.json',manifest)
    files.append(ROOT/'PACKAGE_MANIFEST.json')
    target=ROOT.parent/(ROOT.name+'-artifacts.zip')
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in files: z.write(path,arcname=str(Path('tlab_recsys')/path.relative_to(ROOT)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all('/data/' not in n and '/.venv/' not in n and '/cache/' not in n for n in z.namelist())
    print(f'{target}: {target.stat().st_size/1048576:.2f} MiB, {len(files)} files',flush=True)


if __name__=='__main__': main()
