import tarfile
from pathlib import Path
import csv

def unzip_raw_from_tar(data):
    print(data)
    for day in data:
        print(day)
        folder = "/data/exp/IceCube/2026/internal-system/upgrade-camera/" + str(day).zfill(4)
        base_dir = Path(folder)
        output_dir = Path("/home/jtorresespinosa/jtorresespinosa/UpgradeCamera/UpgradeCameraImageAnalysis/dailyReportRunner/output/"+str(day).zfill(4))
        output_dir.mkdir(parents=True, exist_ok=True)
        print (f"Processing directory: {base_dir}")
        count = 0

        for tar_path in sorted(base_dir.glob("*.tar.gz")):
            # print(f"Processing: {tar_path.name}")

            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                raw_members = [m for m in members if m.name.endswith(".raw")]
                if len(raw_members) != 1:
                    print(f"    Warning: found {len(raw_members)} raw files in {tar_path.name}")
                    continue
                raw_member = raw_members[0]
                # output path:
                out_path = output_dir / Path(raw_member.name).name
                # extract raw only
                with tar.extractfile(raw_member) as f_in, open(out_path, "wb") as f_out:
                    f_out.write(f_in.read())
                
            # # delete tar.gz
            # tar_path.unlink()
            # print(f"    Deleted: {tar_path.name}")
            count += 1

        print(f"\nDone. Processed {count} files.")
    return 0


with open('newDirs.txt', newline='') as csvfile:
    next(csvfile)
    data = list(csv.reader(csvfile))
data = [int(item[0]) for item in data]
unzip_raw_from_tar(data)
