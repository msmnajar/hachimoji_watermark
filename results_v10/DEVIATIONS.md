# Revision 10 — deviations and incidents log

## 2026-10-28 — U input-preparation incident (no change to protocol, code, configurations or frozen artifacts)
- **What happened:** the guarded evaluation for seed 1 stopped (exit 1) while extracting U features for ISIC:
  `OSError: image file is truncated` for `ISIC_5683393.png` (line 1528 of `lap_U_isic.lst`).
- **Cause:** during U image preparation on the laptop, a `prep_v10.py` call was ended by its 165 s time limit while
  writing this file. The resumable script then skipped the existing (partial) file on the next call.
- **Scope:** a full decode of all 3,378 U and 900 C images found this as the only damaged file. The source JPEG is
  intact (1,171,976 bytes). No ISIC result had been computed for any seed; no output of seeds 0 and 2 used the file.
- **Fix (authorized by the author, 2026-10-28):** the partial file was moved to `_to_delete/` on the laptop and
  `/home/claude/data/v10_U_quarantine/` here; the unchanged `prep_v10.py` (sha256 6fadfb97…) regenerated it
  (sha256 028c96eb…, 143,485 bytes). It decodes, and is pixel-identical to an independent conversion of the source
  JPEG on the compute machine. All 3,378 U images decode after the fix.
- **Effect on results:** none intended or expected. The input is the image the frozen procedure specifies. Seed 1 is
  re-run from the start under the same guard; its earlier partial outputs (BraTS, NIH) are overwritten by the re-run.
  The access log records both runs.
