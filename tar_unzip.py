import tarfile
from pathlib import Path

base_dir = Path("data/0107/")   
count = 0

for tar_path in sorted(base_dir.glob("*.tar.gz")):
    print(f"Processing: {tar_path.name}")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        raw_members = [m for m in members if m.name.endswith(".raw")]
        if len(raw_members) != 1:
            print(f"    Warning: found {len(raw_members)} raw files in {tar_path.name}")
            continue
        raw_member = raw_members[0]
        # output path: same directory, just filename
        out_path = base_dir / Path(raw_member.name).name
        # extract raw only
        with tar.extractfile(raw_member) as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"    Extracted: {out_path.name}")
        
    # delete tar.gz
    tar_path.unlink()
    print(f"    Deleted: {tar_path.name}")
    count += 1

print(f"\nDone. Processed {count} files.")
