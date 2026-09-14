"""List or restore a saved Markdown revision; restoration also preserves the current version."""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from storage import atomic_write
import notes
import docx_export


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class",dest="code",required=True)
    parser.add_argument("--revision",help="Exact filename from the revision listing")
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9 _-]+",args.code):
        parser.error("Invalid class code")
    target=notes.notes_path(args.code)
    revisions=target.parent/".revisions"/target.name
    if not args.revision:
        for path in sorted(revisions.glob("*.txt"),reverse=True):
            print(path.name)
        return
    if not re.fullmatch(r"\d{8}_\d{6}_\d{6}\.txt",args.revision):
        parser.error("Use an exact revision filename from the listing")
    revision=revisions/args.revision
    if not revision.is_file(): parser.error("Revision not found")
    text=revision.read_text(encoding="utf-8")
    atomic_write(target,text)
    try:
        docx_export.rebuild(args.code,args.code,text)
    except Exception as error:
        print(f"Markdown restored; Word mirror could not update: {error}")
    print(f"Restored {target}; the previous version is retained in .revisions")


if __name__=="__main__": main()
